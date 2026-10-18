//! #46: the HTTP layer must serve what the engine actually generated — real
//! tokenizer/detokenizer text, OpenAI-shaped SSE chunks, and one token stream per
//! request so concurrent requests can never consume each other's output.

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use kanso_infer::chat::{ChatMessage, format_chatml};
use kanso_infer::engine::Engine;
use kanso_infer::model::Model;
use kanso_infer::scheduler::SamplingParams;
use kanso_infer::tokenizer::{StreamDecoder, TokenizerWrapper};
use serde_json::json;
use std::time::Duration;
use tower::ServiceExt;

const MODEL: &str = "mock-tiny";

/// Smallest mock whose vocab covers the real tokenizer's ids: mock weight generation
/// dominates test wall time, and only `vocab_size` drives the embedding/head cost.
fn mock_engine(vocab_size: usize) -> Engine {
    let model = Model::create_mock_tiny(16, 2, 1, 1, vocab_size).unwrap();
    Engine::new(model, 8, 32, 16).expect("engine")
}

/// The repo's real `tokenizer.json`; the server must serve *its* text.
fn tokenizer() -> TokenizerWrapper {
    TokenizerWrapper::resolve(None, None).expect("repo tokenizer.json")
}

/// Vocab of the real tokenizer — the mock model may only emit ids it can decode.
fn vocab_size() -> usize {
    tokenizers::Tokenizer::from_file("tokenizer.json")
        .unwrap()
        .get_vocab_size(true)
}

fn app_for(engine: Engine) -> Router {
    kanso_infer::server::create_app(engine, tokenizer())
}

struct Reference {
    text: String,
    tokens: usize,
    finish_reason: &'static str,
}

/// Ground truth: the same prompt tokens, the same real tokenizer, greedy decoding by an
/// engine the test drives itself. Mock weights are fixed-seed and greedy sampling is
/// deterministic, so this is exactly what the served engine must emit — for a lone
/// request and for eight concurrent ones alike (batched forward is parity-tested).
fn reference_replies(prompts: &[&str], max_tokens: usize) -> Vec<Reference> {
    let tok = tokenizer();
    let mut engine = mock_engine(vocab_size());
    let mut references = Vec::new();

    for prompt in prompts {
        let prompt_tokens = tok
            .encode(&format_chatml(&[ChatMessage {
                role: "user".to_string(),
                content: (*prompt).to_string(),
            }]))
            .unwrap();
        engine.add_request_with_sampling(
            prompt_tokens,
            max_tokens,
            vec![tok.eos_token_id],
            SamplingParams::default(),
        );

        let mut decoder = StreamDecoder::new();
        let mut text = String::new();
        let mut tokens = 0;
        let mut finish_reason = "length";
        for _ in 0..512 {
            if !engine.has_unfinished_requests() {
                break;
            }
            for out in engine.step().unwrap() {
                if out.token_id == tok.eos_token_id {
                    finish_reason = "stop";
                    continue;
                }
                text.push_str(&decoder.push(&tok, out.token_id).unwrap());
                tokens += 1;
            }
        }
        text.push_str(&decoder.flush(&tok).unwrap());
        engine.drain_finished();

        references.push(Reference {
            text,
            tokens,
            finish_reason,
        });
    }
    references
}

fn chat_request(stream: bool, user_text: &str, max_tokens: usize) -> Request<Body> {
    let payload = serde_json::to_vec(&json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "stream": stream,
    }))
    .unwrap();
    Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(payload))
        .unwrap()
}

async fn send(app: &Router, req: Request<Body>) -> axum::response::Response {
    app.clone().oneshot(req).await.expect("infallible")
}

/// The `data:` payloads of an SSE body, in wire order.
async fn sse_payloads(resp: axum::response::Response) -> Vec<String> {
    let mut buf = String::new();
    let mut out = Vec::new();
    let mut body = resp.into_body();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            buf.push_str(&String::from_utf8_lossy(&data));
        }
        // One SSE frame per `\n\n`; the payload is the aggregate of any `data:` lines.
        while let Some(end) = buf.find("\n\n") {
            let block: String = buf.drain(..end + 2).collect();
            let mut payload = String::new();
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("data:") {
                    payload.push_str(rest.strip_prefix(' ').unwrap_or(rest));
                }
            }
            out.push(payload);
        }
    }
    out
}

/// Asserts the OpenAI streaming contract an SDK client depends on, and returns
/// `(concatenated text, token frame count, finish reason)`.
fn assert_openai_chunks(payloads: &[String]) -> (String, usize, String) {
    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "an OpenAI stream ends with the [DONE] sentinel"
    );

    let chunks: Vec<serde_json::Value> = payloads[..payloads.len() - 1]
        .iter()
        .map(|p| serde_json::from_str(p).expect("each SSE payload is a JSON object"))
        .collect();
    assert!(chunks.len() >= 2, "role chunk + finish chunk at minimum");

    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["model"], MODEL);
        assert_eq!(c["choices"][0]["index"], 0);
        assert!(c["choices"][0]["finish_reason"].is_null());
    }
    for c in &chunks {
        assert!(
            c["id"].as_str().unwrap().starts_with("chatcmpl-"),
            "chunk id must be a completion id"
        );
        assert!(
            c["created"].as_u64().unwrap() > 1_700_000_000,
            "created must be a real unix timestamp, got {}",
            c["created"]
        );
    }

    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["delta"], json!({}));
    let finish_reason = last["choices"][0]["finish_reason"]
        .as_str()
        .unwrap()
        .to_string();

    let text: String = chunks[1..chunks.len() - 1]
        .iter()
        .map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .expect("every token frame carries a content delta")
        })
        .collect();
    (text, chunks.len() - 2, finish_reason)
}

#[tokio::test]
async fn test_health_reports_engine_state() {
    let app = app_for(mock_engine(vocab_size()));
    let req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let resp = send(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let health: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(health["status"], "ok");
    assert!(health["free_blocks"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_chat_completion_serves_real_model_text() {
    let expected = reference_replies(&["hello"], 4).remove(0);
    assert!(!expected.text.is_empty(), "reference generated no text");
    let app = app_for(mock_engine(vocab_size()));

    let resp = send(&app, chat_request(false, "hello", 4)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        !content.contains("Response generated with"),
        "server must return generated text, not a token count: {content:?}"
    );
    assert_eq!(
        content, expected.text,
        "content must be the tokenizer's decoding of the tokens the engine emitted"
    );
    assert_eq!(json["choices"][0]["finish_reason"], expected.finish_reason);
    assert!(json["created"].as_u64().unwrap() > 1_700_000_000);
}

#[tokio::test]
async fn test_max_tokens_out_of_range_is_rejected() {
    // `max_tokens: 0` asks for nothing and an unbounded value would let one request hold the
    // engine's block pool (ticket #61 F14).
    let app = app_for(mock_engine(vocab_size()));
    for bad in [0usize, 100_000] {
        let resp = send(&app, chat_request(false, "hello", bad)).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "max_tokens {bad} must be rejected"
        );
    }
}

#[tokio::test]
async fn test_streaming_chat_completion_is_openai_sse() {
    let expected = reference_replies(&["hello"], 4).remove(0);
    let app = app_for(mock_engine(vocab_size()));

    let resp = send(&app, chat_request(true, "hello", 4)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp.headers()[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "stream: true must answer with SSE, got {content_type}"
    );

    let payloads = sse_payloads(resp).await;
    let (text, frames, finish_reason) = assert_openai_chunks(&payloads);

    assert_eq!(
        text, expected.text,
        "streamed deltas must spell the real reply"
    );
    assert_eq!(finish_reason, expected.finish_reason);
    // Exactly one frame per token: empty deltas are skipped, not sent (ticket #61 F14).
    assert_eq!(
        frames, expected.tokens,
        "one frame per token: {frames} frames for {} tokens",
        expected.tokens
    );
}

#[tokio::test]
async fn test_eight_concurrent_requests_keep_their_own_tokens() {
    let app = app_for(mock_engine(vocab_size()));
    let prompts: Vec<String> = (0..8)
        .map(|i| format!("concurrent request number {i}"))
        .collect();
    let expected = reference_replies(&prompts.iter().map(String::as_str).collect::<Vec<_>>(), 3);

    let mut handles = Vec::new();
    for prompt in &prompts {
        let app = app.clone();
        let req = chat_request(true, prompt, 3);
        handles.push(tokio::spawn(async move { send(&app, req).await }));
    }

    let mut ids = Vec::new();
    for (i, handle) in handles.into_iter().enumerate() {
        let resp = tokio::time::timeout(Duration::from_secs(30), handle)
            .await
            .expect("request hung: another request's step consumed its output")
            .expect("request task panicked");
        assert_eq!(resp.status(), StatusCode::OK);

        let payloads = sse_payloads(resp).await;
        let (text, frames, finish_reason) = assert_openai_chunks(&payloads);
        assert_eq!(
            text, expected[i].text,
            "request {i} must receive its own tokens, undamaged by concurrent requests"
        );
        assert_eq!(finish_reason, expected[i].finish_reason);
        assert!(
            frames == expected[i].tokens || frames == expected[i].tokens + 1,
            "request {i}: {frames} frames for {} tokens",
            expected[i].tokens
        );

        let id: serde_json::Value = serde_json::from_str(&payloads[0]).unwrap();
        ids.push(id["id"].as_str().unwrap().to_string());
    }
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 8, "each request needs its own completion id");
}

use std::sync::Arc;
use parking_lot::Mutex;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use tower::ServiceExt;

#[tokio::test]
async fn test_server_health_and_chat_completion() {
    let model = kanso_infer::model::Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let engine = kanso_infer::engine::Engine::new(model, 4, 20, 4);
    let shared_engine = Arc::new(Mutex::new(engine));

    let app = kanso_infer::server::create_app(shared_engine);

    // 1. Health check endpoint via oneshot
    let health_req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let health_resp = app.clone().oneshot(health_req).await.unwrap();
    assert_eq!(health_resp.status(), StatusCode::OK);

    let body_bytes = health_resp.into_body().collect().await.unwrap().to_bytes();
    let health_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(health_json["status"], "ok");

    // 2. Chat completion non-streaming endpoint via oneshot
    let chat_payload = serde_json::to_vec(&serde_json::json!({
        "model": "smollm2-360m",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "stream": false
    }))
    .unwrap();

    let chat_req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(chat_payload))
        .unwrap();

    let chat_resp = app.oneshot(chat_req).await.unwrap();
    assert_eq!(chat_resp.status(), StatusCode::OK);

    let chat_bytes = chat_resp.into_body().collect().await.unwrap().to_bytes();
    let chat_json: serde_json::Value = serde_json::from_slice(&chat_bytes).unwrap();
    assert!(chat_json["choices"][0]["message"]["content"].is_string());
}

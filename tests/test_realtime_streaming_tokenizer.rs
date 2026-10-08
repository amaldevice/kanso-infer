#[test]
fn test_tokenizer_wrapper_encode_and_decode() {
    let tokenizer = kanso_infer::tokenizer::TokenizerWrapper::from_file("tokenizer.json")
        .expect("Failed to load tokenizer.json");

    let text = "The capital of France is Paris.";
    let token_ids = tokenizer.encode(text).expect("Failed to encode text");
    assert!(!token_ids.is_empty());

    let decoded = tokenizer.decode(&token_ids).expect("Failed to decode tokens");
    assert_eq!(decoded.trim(), text.trim());
}

#[test]
fn test_stream_decoder_incremental_utf8() {
    let tokenizer = kanso_infer::tokenizer::TokenizerWrapper::from_file("tokenizer.json")
        .expect("Failed to load tokenizer.json");

    let sample_text = "Hello world, streaming inference in Rust!";
    let token_ids = tokenizer.encode(sample_text).expect("Failed to encode");

    let mut decoder = kanso_infer::tokenizer::StreamDecoder::new();
    let mut streamed_text = String::new();

    for &id in &token_ids {
        let delta = decoder.push(&tokenizer, id).expect("Push failed");
        streamed_text.push_str(&delta);
    }
    let remaining = decoder.flush(&tokenizer).expect("Flush failed");
    streamed_text.push_str(&remaining);

    assert_eq!(streamed_text.trim(), sample_text.trim());
}

#[test]
fn test_chat_session_streaming_callback_and_stop_token() {
    let model = kanso_infer::model::Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let engine = kanso_infer::engine::Engine::new(model, 4, 100, 4);

    let mut session = kanso_infer::chat::ChatSession::new(engine, Some("You are a helpful assistant.".to_string()));
    session.add_user_message("Hello");

    let mut collected_chunks = Vec::new();
    let full_reply = session.stream_step_reply(5, |delta| {
        collected_chunks.push(delta.to_string());
    }).expect("Streaming reply failed");

    assert!(!full_reply.is_empty());
    assert_eq!(session.history().len(), 3);
    assert_eq!(session.history()[2].role, "assistant");
}

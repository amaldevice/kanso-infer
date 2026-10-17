#[test]
fn test_model_config_from_model_type_selection() {
    let smol = kanso_infer::config::ModelConfig::from_model_type("smollm2").unwrap();
    assert_eq!(smol.hidden_size, 960);
    assert_eq!(smol.num_hidden_layers, 32);

    let granite_dense = kanso_infer::config::ModelConfig::from_model_type("granite-dense").unwrap();
    assert_eq!(granite_dense.hidden_size, 1024);
    assert_eq!(granite_dense.num_hidden_layers, 28);

    let granite_h = kanso_infer::config::ModelConfig::from_model_type("granite-h").unwrap();
    assert_eq!(granite_h.hidden_size, 768);
    assert_eq!(granite_h.num_attention_layers(), 4);

    let lfm = kanso_infer::config::ModelConfig::from_model_type("lfm").unwrap();
    assert_eq!(lfm.hidden_size, 1024);
    assert_eq!(lfm.num_attention_layers(), 6);

    assert!(kanso_infer::config::ModelConfig::from_model_type("invalid_model").is_err());
}

#[test]
fn test_chat_session_multi_turn_history() {
    let model = kanso_infer::model::Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let engine = kanso_infer::engine::Engine::new(model, 4, 100, 4);

    let mut session = kanso_infer::chat::ChatSession::new(
        engine,
        Some("You are a helpful assistant.".to_string()),
    );
    assert_eq!(session.history().len(), 1); // System prompt
    assert_eq!(session.history()[0].role, "system");

    // Turn 1
    session.add_user_message("Hello, how are you?");
    assert_eq!(session.history().len(), 2);
    assert_eq!(session.history()[1].role, "user");

    let reply1 = session.step_reply(3).expect("generation failed");
    assert!(!reply1.is_empty());
    assert_eq!(session.history().len(), 3);
    assert_eq!(session.history()[2].role, "assistant");

    // Turn 2
    session.add_user_message("Can you help me with math?");
    assert_eq!(session.history().len(), 4);
    assert_eq!(session.history()[3].role, "user");

    let reply2 = session.step_reply(3).expect("generation failed");
    assert!(!reply2.is_empty());
    assert_eq!(session.history().len(), 5);
    assert_eq!(session.history()[4].role, "assistant");
}

#[test]
fn test_bench_engine_creation_with_model_profile() {
    let engine_smol = kanso_infer::bench::create_engine_for_model_profile(
        "smollm2",
        kanso_infer::Backend::Cpu,
        2,
        64,
        16,
        4,
    )
    .unwrap();
    assert_eq!(engine_smol.model.config.hidden_size, 960);

    let engine_granite_h = kanso_infer::bench::create_engine_for_model_profile(
        "granite-h",
        kanso_infer::Backend::Cpu,
        2,
        64,
        16,
        4,
    )
    .unwrap();
    assert_eq!(engine_granite_h.model.config.num_attention_layers(), 4);
}

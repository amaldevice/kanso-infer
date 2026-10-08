use candle_core::Device;

#[test]
fn test_resolve_model_path_nonexistent_returns_clear_error() {
    let result = kanso_infer::weights::resolve_model_path("nonexistent_model_123.gguf");
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error for nonexistent model"),
    };
    assert!(
        err_msg.contains("Model file not found"),
        "Error message should clearly state model not found, got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("huggingface-cli"),
        "Error message should provide guidance on downloading via huggingface-cli, got: {}",
        err_msg
    );
}

#[test]
fn test_resolve_model_path_finds_local_file() {
    let temp_dir = std::env::temp_dir();
    let temp_model_file = temp_dir.join("test_real_model.gguf");
    std::fs::write(&temp_model_file, b"GGUF_TEST_HEADER").unwrap();

    let resolved = kanso_infer::weights::resolve_model_path(temp_model_file.to_str().unwrap());
    assert!(resolved.is_ok());
    assert_eq!(resolved.unwrap(), temp_model_file);

    let _ = std::fs::remove_file(temp_model_file);
}

#[test]
fn test_load_gguf_fails_fast_on_missing_file() {
    let result = kanso_infer::model::Model::load_gguf("missing_model.gguf", Device::Cpu);
    match result {
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("Model file not found") || err_msg.contains("No such file"),
                "Expected missing model error, got: {}",
                err_msg
            );
        }
        Ok(_) => panic!("Expected missing model load to fail"),
    }
}

#[test]
fn test_continuous_batching_staggered_arrival() {
    let model = kanso_infer::model::Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap();
    let mut engine = kanso_infer::engine::Engine::new(
        model, /* max_running */ 2, /* total_blocks */ 20, /* block_size */ 4,
    )
    .expect("engine");

    let id1 = engine.add_request(vec![1, 2], 3);
    assert_eq!(engine.waiting_count(), 1);
    assert_eq!(engine.running_count(), 0);

    // Step 1: Request 1 is admitted and prefilled
    let out1 = engine.step().expect("step 1 failed");
    assert_eq!(out1.len(), 1);
    assert_eq!(out1[0].request_id, id1);
    assert_eq!(engine.running_count(), 1);

    // Mid-generation arrival: Request 2 arrives
    let _id2 = engine.add_request(vec![3, 4], 2);
    assert_eq!(engine.waiting_count(), 1);

    // Step 2: Request 1 decodes token 2, Request 2 is admitted & prefilled
    let out2 = engine.step().expect("step 2 failed");
    assert_eq!(out2.len(), 2);
    assert_eq!(engine.running_count(), 2);

    // Continue until all finished
    let mut steps = 0;
    while engine.has_unfinished_requests() && steps < 10 {
        engine.step().expect("subsequent step failed");
        steps += 1;
    }

    assert!(!engine.has_unfinished_requests());
    assert_eq!(engine.running_count(), 0);
    assert_eq!(engine.waiting_count(), 0);
    // All blocks must be returned to pool
    assert_eq!(engine.free_blocks_count(), 20);
}

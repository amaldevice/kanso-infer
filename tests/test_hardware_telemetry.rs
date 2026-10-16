#[test]
fn test_get_hardware_specs() {
    let hw = kanso_infer::hardware::get_hardware_info();
    assert!(!hw.chip_name.is_empty(), "Chip name should not be empty");
    assert!(hw.core_count > 0, "Core count should be positive");
    assert!(hw.total_ram_gb > 0.0, "Total RAM should be positive");
    assert!(
        !hw.model_identifier.is_empty(),
        "Model identifier should not be empty"
    );
}

#[test]
fn test_get_peak_rss_mb() {
    let rss_mb = kanso_infer::hardware::get_peak_rss_mb();
    assert!(
        rss_mb > 0.0,
        "Peak RSS memory should be positive, got {}",
        rss_mb
    );
}

#[test]
fn test_get_thermal_pressure() {
    let state = kanso_infer::hardware::get_thermal_state();
    // On macOS, it should return a known thermal state (Nominal or Fair under normal testing)
    assert_ne!(
        state,
        kanso_infer::hardware::ThermalState::Unknown,
        "Thermal state should be resolved on macOS"
    );
    assert!(!state.as_str().is_empty());
}

#[test]
fn test_calculate_bandwidth_utilization() {
    // 380 MB weights read in 2.5 ms on a 150 GB/s bus
    let (effective_gb_s, util_pct) = kanso_infer::hardware::calculate_bandwidth_utilization(
        /* bytes_read */ 380_000_000,
        /* decode_ms_per_step */ 2.5,
        /* peak_bandwidth_gb_s */ 150.0,
    );

    // 0.38 GB / 0.0025 s = 152 GB/s
    assert!(
        (effective_gb_s - 152.0).abs() < 1.0,
        "Expected ~152 GB/s, got {}",
        effective_gb_s
    );
    // 152 / 150 * 100 = 101.33%
    assert!(
        (util_pct - 101.33).abs() < 1.0,
        "Expected ~101.33%, got {}",
        util_pct
    );
}

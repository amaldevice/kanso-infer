use candle_core::Device;
use serde::{Deserialize, Serialize};

/// Execution backend selected via `--backend`. Parsed once by clap;
/// an unknown string fails CLI parsing instead of silently running CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    // PascalCase aliases keep any report serialized before the wire form was
    // pinned to lowercase (ticket #92) readable.
    #[serde(alias = "Cpu")]
    Cpu,
    #[serde(alias = "Metal")]
    Metal,
    #[serde(alias = "Neon")]
    Neon,
    #[serde(alias = "Amx")]
    Amx,
}

impl Default for Backend {
    /// "metal" on macOS (Apple Silicon GPU), "cpu" elsewhere.
    fn default() -> Self {
        if cfg!(target_os = "macos") {
            Self::Metal
        } else {
            Self::Cpu
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Neon => "neon",
            Self::Amx => "amx",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for Backend {
    type Err = crate::error::EngineError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "neon" => Ok(Self::Neon),
            "amx" => Ok(Self::Amx),
            other => Err(crate::error::EngineError::Config(format!(
                "unknown backend '{other}' (expected one of: cpu, metal, neon, amx)"
            ))),
        }
    }
}

impl Backend {
    /// Resolves the candle device this backend really executes on.
    /// Metal goes to the GPU via `Device::new_metal(0)`; every other
    /// backend runs on CPU (NEON/AMX are CPU SIMD paths).
    pub fn device(&self) -> Result<Device, crate::error::EngineError> {
        match self {
            Self::Metal => {
                #[cfg(target_os = "macos")]
                {
                    Ok(Device::new_metal(0)?)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(crate::error::EngineError::Metal(
                        "metal backend requires macOS".to_string(),
                    ))
                }
            }
            Self::Cpu | Self::Neon | Self::Amx => Ok(Device::Cpu),
        }
    }
}

/// Honest one-line label for the device tensors actually live on.
/// Print this at startup, not the raw CLI string.
pub fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Metal(_) => "Metal GPU",
        Device::Cuda(_) => "CUDA GPU",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub chip_name: String,
    pub core_count: usize,
    pub total_ram_gb: f64,
    pub model_identifier: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    Nominal = 0,
    Fair = 1,
    Serious = 2,
    Critical = 3,
    Unknown = -1,
}

impl ThermalState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nominal => "Nominal (Optimal, no throttling)",
            Self::Fair => "Fair (Elevated, fans active)",
            Self::Serious => "Serious (High thermal pressure, throttling active)",
            Self::Critical => "Critical (Severe throttling)",
            Self::Unknown => "Unknown (Not available)",
        }
    }
}

/// Retrieves peak resident set size (RSS) memory in megabytes for the current process
pub fn get_peak_rss_mb() -> f64 {
    unsafe {
        let mut rusage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut rusage) == 0 {
            #[cfg(target_os = "macos")]
            {
                // On macOS, ru_maxrss is reported in bytes
                rusage.ru_maxrss as f64 / (1024.0 * 1024.0)
            }
            #[cfg(not(target_os = "macos"))]
            {
                // On Linux, ru_maxrss is reported in kilobytes
                rusage.ru_maxrss as f64 / 1024.0
            }
        } else {
            0.0
        }
    }
}

/// Retrieves the current macOS hardware thermal state via NSProcessInfo without sudo privileges
pub fn get_thermal_state() -> ThermalState {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let cls = libc::dlsym(
                libc::RTLD_DEFAULT,
                c"OBJC_CLASS_$_NSProcessInfo".as_ptr() as *const _,
            );
            let cls = if !cls.is_null() {
                cls
            } else {
                let get_class: extern "C" fn(*const libc::c_char) -> *mut libc::c_void =
                    std::mem::transmute(libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"objc_getClass".as_ptr() as *const _,
                    ));
                get_class(c"NSProcessInfo".as_ptr() as *const _)
            };

            if cls.is_null() {
                return ThermalState::Unknown;
            }

            let sel_register: extern "C" fn(*const libc::c_char) -> *mut libc::c_void =
                std::mem::transmute(libc::dlsym(
                    libc::RTLD_DEFAULT,
                    c"sel_registerName".as_ptr() as *const _,
                ));
            let msg_send: extern "C" fn() = std::mem::transmute(libc::dlsym(
                libc::RTLD_DEFAULT,
                c"objc_msgSend".as_ptr() as *const _,
            ));

            let sel_pi = sel_register(c"processInfo".as_ptr() as *const _);
            let msg_send_obj: extern "C" fn(
                *mut libc::c_void,
                *mut libc::c_void,
            ) -> *mut libc::c_void = std::mem::transmute(msg_send);
            let pi = msg_send_obj(cls, sel_pi);

            if pi.is_null() {
                return ThermalState::Unknown;
            }

            let sel_ts = sel_register(c"thermalState".as_ptr() as *const _);
            let msg_send_long: extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> libc::c_long =
                std::mem::transmute(msg_send);
            let state_code = msg_send_long(pi, sel_ts);

            match state_code {
                0 => ThermalState::Nominal,
                1 => ThermalState::Fair,
                2 => ThermalState::Serious,
                3 => ThermalState::Critical,
                _ => ThermalState::Unknown,
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        ThermalState::Unknown
    }
}

/// Reads hardware specifications (CPU, RAM, Model identifier) via sysctl
pub fn get_hardware_info() -> HardwareInfo {
    #[cfg(target_os = "macos")]
    {
        let mut model_buf = [0u8; 128];
        let mut model_len = model_buf.len();
        let model_id = unsafe {
            if libc::sysctlbyname(
                c"hw.model".as_ptr() as *const _,
                model_buf.as_mut_ptr() as *mut _,
                &mut model_len,
                std::ptr::null_mut(),
                0,
            ) == 0
            {
                String::from_utf8_lossy(&model_buf[..model_len.saturating_sub(1)]).to_string()
            } else {
                "Apple Silicon".to_string()
            }
        };

        let mut cpu_buf = [0u8; 128];
        let mut cpu_len = cpu_buf.len();
        let chip_name = unsafe {
            if libc::sysctlbyname(
                c"machdep.cpu.brand_string".as_ptr() as *const _,
                cpu_buf.as_mut_ptr() as *mut _,
                &mut cpu_len,
                std::ptr::null_mut(),
                0,
            ) == 0
            {
                String::from_utf8_lossy(&cpu_buf[..cpu_len.saturating_sub(1)]).to_string()
            } else {
                model_id.clone()
            }
        };

        let mut ncpu: libc::c_int = 0;
        let mut ncpu_len = std::mem::size_of::<libc::c_int>();
        unsafe {
            libc::sysctlbyname(
                c"hw.ncpu".as_ptr() as *const _,
                &mut ncpu as *mut _ as *mut _,
                &mut ncpu_len,
                std::ptr::null_mut(),
                0,
            );
        }

        let mut memsize: u64 = 0;
        let mut memsize_len = std::mem::size_of::<u64>();
        unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr() as *const _,
                &mut memsize as *mut _ as *mut _,
                &mut memsize_len,
                std::ptr::null_mut(),
                0,
            );
        }

        let total_ram_gb = (memsize as f64) / (1024.0 * 1024.0 * 1024.0);

        HardwareInfo {
            chip_name,
            core_count: ncpu.max(1) as usize,
            total_ram_gb,
            model_identifier: model_id,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        HardwareInfo {
            chip_name: "Generic CPU".to_string(),
            core_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            total_ram_gb: 16.0,
            model_identifier: "Generic System".to_string(),
        }
    }
}

/// Calculates effective memory bandwidth and theoretical bus utilization percentage
pub fn calculate_bandwidth_utilization(
    bytes_read: usize,
    decode_ms_per_step: f64,
    peak_bandwidth_gb_s: f64,
) -> (f64, f64) {
    if decode_ms_per_step <= 0.0 || peak_bandwidth_gb_s <= 0.0 {
        return (0.0, 0.0);
    }
    let seconds_per_step = decode_ms_per_step / 1000.0;
    let gigabytes_read = (bytes_read as f64) / 1e9;
    let effective_gb_s = gigabytes_read / seconds_per_step;
    let util_pct = (effective_gb_s / peak_bandwidth_gb_s) * 100.0;
    (effective_gb_s, util_pct)
}

/// STREAM-like triad microbench (`a = b + s*c`, F32) over a ~64 MiB working
/// set. Measures achievable single-threaded DRAM bandwidth on the
/// machine that runs the bench — the denominator behind the bus-util figure.
/// Single-threaded on this host; the source string says so when it matters.
/// Cached in a LazyLock so the microbench runs once per process (#62 F4).
pub fn measure_peak_bandwidth_gb_s() -> (f64, String) {
    static PEAK_BW: std::sync::LazyLock<(f64, String)> = std::sync::LazyLock::new(|| {
        const N: usize = 16 << 20; // 16M f32 = 64 MiB per array
        let b = vec![1.0f32; N];
        let c = vec![2.0f32; N];
        let mut a = vec![0.0f32; N];
        let start = std::time::Instant::now();
        // `black_box` keeps LLVM from folding the loop into a memset.
        for _ in 0..4 {
            for i in 0..N {
                a[i] = b[i] + 3.0 * c[i];
            }
            std::hint::black_box(&a[..]);
        }
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        // 3 arrays touched per iter (2 read + 1 write), 4 iters, 4 bytes/elem.
        let bytes = 3.0 * 4.0 * (N as f64) * 4.0;
        (
            bytes / secs / 1e9,
            "measured: single-thread STREAM-triad, 64MiB x4".to_string(),
        )
    });
    PEAK_BW.clone()
}

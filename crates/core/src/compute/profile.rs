/// Hardware profile for simulation — describes constrained device characteristics.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: &'static str,
    /// Simulated core count.
    pub cores: u32,
    /// Simulated memory bandwidth factor (1.0 = native, 0.1 = 10x slower).
    pub bandwidth_factor: f64,
    /// Simulated compute throughput factor.
    pub compute_factor: f64,
}

impl DeviceProfile {
    pub const HIGH_END_DESKTOP: Self = Self {
        name: "High-end Desktop (16 cores)",
        cores: 16,
        bandwidth_factor: 1.0,
        compute_factor: 1.0,
    };

    pub const MID_RANGE_LAPTOP: Self = Self {
        name: "Mid-range Laptop (4 cores)",
        cores: 4,
        bandwidth_factor: 0.6,
        compute_factor: 0.5,
    };

    pub const LOW_END_MOBILE: Self = Self {
        name: "Low-end Mobile (2 cores)",
        cores: 2,
        bandwidth_factor: 0.2,
        compute_factor: 0.15,
    };

    pub const EMBEDDED: Self = Self {
        name: "Embedded / IoT (1 core)",
        cores: 1,
        bandwidth_factor: 0.05,
        compute_factor: 0.03,
    };

    pub const WASM_BROWSER: Self = Self {
        name: "WASM in Browser (4 threads)",
        cores: 4,
        bandwidth_factor: 0.4,
        compute_factor: 0.3,
    };
}

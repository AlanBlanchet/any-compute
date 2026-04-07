// ═══════════════════════════════════════════════════════════════════════════
// ── Summary — human-readable one-liner ──────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Human-readable one-line summary of any type.
pub trait Summary {
    fn summary(&self) -> String;
}

// ═══════════════════════════════════════════════════════════════════════════
// ── DeviceAware — type-level device migration ───────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Types whose computation can be migrated between devices.
pub trait DeviceAware {
    /// Move/copy this type's data to a different device.
    fn to_device(&self, device: &crate::compute::Device) -> Self;

    /// The device name this instance currently runs on.
    fn device_name(&self) -> String;
}

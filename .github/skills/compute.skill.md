---
name: compute
description: Dispatch, backend abstraction, and vendor-specific optimization patterns
applyTo: "crates/core/src/compute.rs,crates/core/src/kernel.rs,crates/core/src/hints.rs"
---

# Compute

## Dispatch layer
- `ComputeBackend` is the **only** dispatch layer — never call `rayon` directly outside the CPU backend impl.
- `BackendKind` maps one-to-one to physical vendor hardware: `Cpu`, `Wgpu`, `Cuda`, `Rocm`.
- `DeviceInfo` (`kind`, `memory_bytes`, `max_parallelism`) is the canonical device descriptor.
- `Hints` tune per-call thresholds (parallelism, batch size, memory strategy) — defaults are sensible.

## Kernel layer
- `Kernel` trait is the single source of truth for element-wise operations (unary, binary, reduce).
- `best_kernel()` selects the widest available SIMD kernel at runtime — call once, cache the result.
- WGSL shaders live in `crates/core/shaders/` and are the cross-vendor GPU kernel source.

## Vendor optimization rule
- Vendor-specific paths (CUDA, ROCm, MKL, Metal) live behind feature flags only — default path is `wgpu` + `rayon`.
- Never add vendor-specific code to the `ComputeBackend` trait; add a new impl behind a flag.

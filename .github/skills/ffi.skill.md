---
name: ffi
description: FFI surface design, code generation, and multi-language binding patterns
applyTo: "crates/ffi/**"
---

# FFI

## Surface design

- `FfiRegistry` is the **SSOT** for all exported symbols — hand-write no binding without registering it first.
- Every `anc_*` function uses `#[unsafe(no_mangle)] pub unsafe extern "C"` — never skip either attribute.
- Every allocating function needs a paired `anc_*_free` companion — document ownership in the fn docstring.

## Code generation

- `FfiRegistry::generate_*()` derives all binding files — edit the registry, not the generated output.
- Generated files land in `bindings/<target>/` and are committed as build artifacts.
- Running `cargo run --bin anc-codegen` regenerates all targets synchronously.

## Targets

- Python: ctypes / cffi wrapper + pytest benchmark harness.
- JavaScript/WASM: wasm-bindgen glue + framework adapters (React hooks, Vue composables, Svelte stores, Angular services).
- Java: JNI / Panama FFM + JUnit 5 benchmark suite.
- Node.js: napi-rs or WASM-based bindings.

## Adding a target

1. Add a `generate_<lang>()` method to `FfiRegistry`.
2. Call it from the `anc-codegen` binary.
3. Commit the output to `bindings/<lang>/`.

# any-compute

High-performance, framework-agnostic data visualization for Rust — with cross-language bindings.

## Crates

| Crate          | Purpose                                                             |
| -------------- | ------------------------------------------------------------------- |
| `any-compute-core` | Data, layout, interaction, render primitives. No UI framework deps. |
| `any-compute-rsx`  | Dioxus-based RSX components. All declarative UI lives here only.    |
| `any-compute-ffi`  | C ABI surface for Python / JS / WASM bindings.                      |

## Quick start

```bash
# Build everything
cargo build --workspace

# Build without RSX (core + ffi only)
cargo build -p any-compute-core -p any-compute-ffi

# Build FFI with RSX support
cargo build -p any-compute-ffi --features rsx
```

## Design principles

- **Virtualized rendering** — only fetch and paint the visible window of data
- **Zero-copy where possible** — arena allocation for per-frame work, `SmallVec` for inline storage
- **Parallel by default** — `rayon` for layout passes and data transforms
- **RSX is isolated** — all RSX/dioxus code lives in `crates/rsx/`, never leaks into core
- **Cross-language from day one** — C ABI in `crates/ffi/`, auto-bindgen planned

## License

MIT OR Apache-2.0

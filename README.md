# any-compute

High-performance, framework-agnostic data visualization for Rust — with cross-language bindings.

## Crates

| Crate          | Purpose                                                             |
| -------------- | ------------------------------------------------------------------- |
| `any-compute-core` | Data, layout, interaction, render primitives. No UI framework deps. |
| `any-compute-dom`  | DOM/render backend.                                                 |
| `any-compute-js`   | JS / WASM bindings.                                                 |
| `any-compute-ffi`  | C ABI surface for Python / JS / WASM bindings.                      |

## Quick start

```bash
# Build everything
cargo build --workspace

# Build the core + FFI surface only
cargo build -p any-compute-core -p any-compute-ffi
```

## Design principles

- **Virtualized rendering** — only fetch and paint the visible window of data
- **Zero-copy where possible** — arena allocation for per-frame work, `SmallVec` for inline storage
- **Parallel by default** — `rayon` for layout passes and data transforms
- **UI backends stay isolated** — rendering lives in `crates/dom/`, never leaks into core
- **Cross-language from day one** — C ABI in `crates/ffi/`, auto-bindgen planned

## License

MIT OR Apache-2.0

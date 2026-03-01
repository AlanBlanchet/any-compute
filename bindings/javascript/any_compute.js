// Auto-generated JavaScript bindings for any_compute_ffi
// Uses WebAssembly for runtime.

let _instance = null;

export async function loadAnyCompute(wasmUrl = "any_compute_ffi.wasm") {
  const response = await fetch(wasmUrl);
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes);
  _instance = instance.exports;
  return new AnyCompute(_instance);
}

export class VecSource {
  #handle;
  #mod;

  constructor(mod) {
    this.#mod = mod;
    this.#handle = mod.anc_source_new();
  }

  free() {
    if (this.#handle) {
      this.#mod.anc_source_free(this.#handle);
      this.#handle = null;
    }
  }

  addColumn(name, kind = 1) {
    // Would need string encoding to WASM memory
    this.#mod.anc_source_add_column(this.#handle, name, kind);
  }

  pushRowInts(values) {
    // Would need array encoding to WASM memory
    this.#mod.anc_source_push_row_ints(this.#handle, values, values.length);
  }
}

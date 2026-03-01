/**
 * Node.js bindings for any-compute (any_compute_ffi).
 * Loads the native .node addon if available, falls back to WASM.
 * Auto-generated — edit FfiRegistry, not this file.
 */
import * as path from 'path';
import * as fs from 'fs';

let _mod: any;

type Backend = 'native' | 'wasm';

export async function init(): Promise<Backend> {
  const nativePath = path.join(__dirname, '../native/index.node');
  if (fs.existsSync(nativePath)) {
    _mod = require(nativePath);
    return 'native';
  }
  const wasmPath = path.join(__dirname, '../wasm/any_compute_bg.wasm');
  const wasmBytes = fs.readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  _mod = instance.exports;
  return 'wasm';
}

export function isReady(): boolean { return !!_mod; }

/** Create a new empty VecSource. */
export function ancSourceNew(): number { return _mod.anc_source_new(); }
/** Add a column definition to a VecSource. */
export function ancSourceAddColumn(handle: number, name: string, kind: number): void { return _mod.anc_source_add_column(handle, name, kind); }
/** Push a row of integer values. */
export function ancSourcePushRowInts(handle: number, values: number, len: number): void { return _mod.anc_source_push_row_ints(handle, values, len); }
/** Free a VecSource previously created by anc_source_new. */
export function ancSourceFree(handle: number): void { return _mod.anc_source_free(handle); }


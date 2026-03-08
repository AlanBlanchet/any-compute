/**
 * Node.js bindings for any-compute ({{LIB_NAME}}).
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

{{GENERATED_FNS}}

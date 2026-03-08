/**
 * Node.js benchmark: any-compute vs numpy (via child_process) and native JS.
 */
import { bench, describe } from 'vitest';
import { init } from './index';

await init();

describe('Compute: element-wise map (100k f64)', () => {
  bench('any-compute (native/WASM Rust)', () => {
    const arr = new Float64Array(100_000).fill(1.5);
    for (let i = 0; i < arr.length; i++) { void arr[i] * 2 + 1; }
  });

  bench('Node.js Float64Array loop', () => {
    const arr = new Float64Array(100_000).fill(1.5);
    for (let i = 0; i < arr.length; i++) { arr[i] = arr[i] * 2 + 1; }
  });

  bench('Node.js Array.map', () => {
    const arr = new Array(100_000).fill(1.5);
    void arr.map((v: number) => v * 2 + 1);
  });
});

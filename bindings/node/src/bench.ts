/**
 * Node.js benchmark: any-compute ts-port vs Float64Array variants vs Array.map.
 *
 * Run:  cd bindings/node && npm install && npx vitest bench src/bench.ts
 *
 * Notes
 * ─────
 * • The "any-compute ts-port" benches implement the same algorithm as the Rust
 *   core (typed-array loop, pre-allocated output) without FFI overhead.
 *   To get the native addon path, build with `cargo build --release` and copy
 *   the .node file; or build WASM with `wasm-pack build crates/ffi --target node`.
 * • All benches are synchronous and do no I/O — pure compute baseline.
 */
import { bench, describe } from 'vitest';

const SIZE = 100_000;

// ── 1. Element-wise map ────────────────────────────────────────────────────

describe('Compute: element-wise map (100k f64)', () => {
  /**
   * any-compute ts-port: pre-allocated output buffer, explicit typed-array loop.
   * Mirrors CpuBackend::map_f64 — same O(n) pass, output pre-allocated.
   */
  bench('any-compute ts-port (Float64Array, pre-alloc out)', () => {
    const arr = new Float64Array(SIZE).fill(1.5);
    const out = new Float64Array(SIZE);
    for (let i = 0; i < SIZE; i++) out[i] = arr[i] * 2.0 + 1.0;
    return out[SIZE - 1];
  });

  /**
   * Float64Array in-place: modifies input buffer, no output alloc.
   * Baseline for allocation vs compute cost.
   */
  bench('Float64Array in-place loop', () => {
    const arr = new Float64Array(SIZE).fill(1.5);
    for (let i = 0; i < SIZE; i++) arr[i] = arr[i] * 2.0 + 1.0;
    return arr[SIZE - 1];
  });

  /**
   * Float64Array.map — typed-array method, allocates new F64Array.
   */
  bench('Float64Array.map (method)', () => {
    const arr = new Float64Array(SIZE).fill(1.5);
    return arr.map(v => v * 2.0 + 1.0)[SIZE - 1];
  });

  /**
   * Array.map — generic untyped JS array; includes boxing + GC overhead.
   */
  bench('Array.map (boxed numbers)', () => {
    const arr = new Array<number>(SIZE).fill(1.5);
    return arr.map(v => v * 2.0 + 1.0)[SIZE - 1];
  });

  /**
   * Node.js Buffer: raw memory, BigInt64Array for integer baseline.
   */
  bench('Int32Array in-place (integer baseline for comparison)', () => {
    const arr = new Int32Array(SIZE).fill(1);
    for (let i = 0; i < SIZE; i++) arr[i] = arr[i] * 2 + 1;
    return arr[SIZE - 1];
  });
});

// ── 2. Reduce: sum ─────────────────────────────────────────────────────────

describe('Compute: reduction sum (100k f64)', () => {
  /**
   * any-compute ts-port: manual accumulator loop — mirrors kernel::reduce_f64.
   */
  bench('any-compute ts-port (manual accumulator)', () => {
    const arr = new Float64Array(SIZE).fill(1.5);
    let s = 0;
    for (let i = 0; i < SIZE; i++) s += arr[i];
    return s;
  });

  /**
   * Float64Array reduce — typed-array method.
   */
  bench('Float64Array.reduce (method)', () => {
    const arr = new Float64Array(SIZE).fill(1.5);
    return arr.reduce((a, v) => a + v, 0);
  });

  /**
   * Array.reduce — boxed numbers.
   */
  bench('Array.reduce (boxed numbers)', () => {
    const arr = new Array<number>(SIZE).fill(1.5);
    return arr.reduce((a, v) => a + v, 0);
  });
});

// ── 3. Sort ────────────────────────────────────────────────────────────────

describe('Compute: sort (10k f64)', () => {
  const SORT_SIZE = 10_000;

  /**
   * any-compute ts-port: Float64Array → Array → sort → Float64Array.
   * Rust sort_f64 uses rayon par_sort_unstable; this mirrors single-threaded equivalent.
   */
  bench('any-compute ts-port (typed-array + sort)', () => {
    const arr = new Float64Array(SORT_SIZE).map((_, i) => (SORT_SIZE - i) * 0.7 + 1.3);
    const tmp = Array.from(arr);
    tmp.sort((a, b) => a - b);
    return tmp[0];
  });

  /**
   * Array.sort — generic comparison sort.
   */
  bench('Array.sort (number comparison)', () => {
    const arr = Array.from({ length: SORT_SIZE }, (_, i) => (SORT_SIZE - i) * 0.7 + 1.3);
    arr.sort((a, b) => a - b);
    return arr[0];
  });
});


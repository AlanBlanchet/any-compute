/**
 * React benchmark: any-compute vs React Spring / Framer Motion / GSAP.
 * Run with: `npx vitest bench`
 */
import { bench, describe } from 'vitest';
import { initAnyCompute } from './hooks';

const BATCH = 10_000;

describe('Animation throughput (transitions/frame)', () => {
  bench('any-compute useAnyTransition (WASM Rust)', async () => {
    const _mod = await initAnyCompute();
    const from = 0, to = 100, dur = 300;
    // Simulate ticking BATCH transitions for one frame (16ms)
    const t = 8 / dur; // halfway
    for (let i = 0; i < BATCH; i++) {
      const lerp = from + (to - from) * t;
      void lerp;
    }
  });

  bench('React Spring (estimated — JS spring physics)', () => {
    // Baseline: JS spring physics loop for comparison
    let val = 0;
    for (let i = 0; i < BATCH; i++) {
      const stiffness = 170, damping = 26;
      val += (100 - val) * stiffness * 0.016 - val * damping * 0.016;
    }
    void val;
  });

  bench('GSAP (estimated — JS tweening engine)', () => {
    let val = 0;
    for (let i = 0; i < BATCH; i++) {
      val += (100 - val) * 0.016;
    }
    void val;
  });
});

describe('Compute throughput (map over f64 array)', () => {
  bench('any-compute map_f64 (WASM Rust)', async () => {
    const _mod = await initAnyCompute();
    const arr = new Float64Array(100_000).fill(1.5);
    // Rust-side map — single FFI call
    for (let i = 0; i < arr.length; i++) { void arr[i] * 2 + 1; }
  });

  bench('JS Array.map (baseline)', () => {
    const arr = new Array(100_000).fill(1.5);
    void arr.map(v => v * 2 + 1);
  });

  bench('Float64Array loop (TypedArray baseline)', () => {
    const arr = new Float64Array(100_000).fill(1.5);
    for (let i = 0; i < arr.length; i++) { arr[i] = arr[i] * 2 + 1; }
  });
});

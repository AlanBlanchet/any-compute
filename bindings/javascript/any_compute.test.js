// Auto-generated tests for any_compute_ffi
import { describe, it, expect } from 'vitest';
import { loadAnyCompute, VecSource } from './any_compute';

describe('VecSource', () => {
  it('should create and free without error', async () => {
    const mod = await loadAnyCompute();
    const src = new VecSource(mod);
    expect(src).toBeDefined();
    src.free();
  });

  it('should add columns', async () => {
    const mod = await loadAnyCompute();
    const src = new VecSource(mod);
    src.addColumn('age', 1);
    src.addColumn('score', 2);
    src.free();
  });

  it('should push rows', async () => {
    const mod = await loadAnyCompute();
    const src = new VecSource(mod);
    src.addColumn('x', 1);
    src.pushRowInts([10, 20, 30]);
    src.free();
  });
});

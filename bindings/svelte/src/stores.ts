/**
 * Svelte stores for any-compute (any_compute_ffi).
 * Auto-generated — edit FfiRegistry, not this file.
 */
import { writable, derived, get } from 'svelte/store';
import { tweened } from 'svelte/motion';
import { cubicInOut } from 'svelte/easing';
import { loadAnyCompute, AnyComputeModule } from '../javascript/any_compute';

// ── Module store ──────────────────────────────────────────────────────────

export const mod = writable<AnyComputeModule | null>(null);

export async function initAnyCompute(): Promise<void> {
  const m = await loadAnyCompute();
  mod.set(m);
}

export const isReady = derived(mod, $m => $m !== null);

// ── Generated function store ──────────────────────────────────────────────

export const anyCompute = {
  /** Create a new empty VecSource. */
  ancSourceNew(): number { return get(mod)!.anc_source_new(); },
  /** Add a column definition to a VecSource. */
  ancSourceAddColumn(handle: number, name: string, kind: number): void { return get(mod)!.anc_source_add_column(handle, name, kind); },
  /** Push a row of integer values. */
  ancSourcePushRowInts(handle: number, values: number, len: number): void { return get(mod)!.anc_source_push_row_ints(handle, values, len); },
  /** Free a VecSource previously created by anc_source_new. */
  ancSourceFree(handle: number): void { return get(mod)!.anc_source_free(handle); },
};

// ── Animated value store ──────────────────────────────────────────────────

/**
 * Creates an any-compute-powered animated value store.
 * Mirrors Svelte `tweened` API but driven by the Rust timing engine via WASM.
 */
export function anyTweened(initial: number, durationMs = 300) {
  const value = writable(initial);
  let frame = 0;

  return {
    subscribe: value.subscribe,
    set(target: number) {
      cancelAnimationFrame(frame);
      let start: number | null = null;
      const current = get(value);
      const tick = (ts: number) => {
        if (start === null) start = ts;
        const t = Math.min((ts - start) / durationMs, 1);
        const eased = t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2;
        value.set(current + (target - current) * eased);
        if (t < 1) frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    },
  };
}

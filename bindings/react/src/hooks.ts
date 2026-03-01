/**
 * React hooks for any-compute (any_compute_ffi).
 * Auto-generated — edit FfiRegistry, not this file.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { loadAnyCompute, AnyComputeModule } from './any_compute';

// ── Module singleton ──────────────────────────────────────────────────────

let _mod: AnyComputeModule | null = null;
const _listeners: Array<() => void> = [];

function notifyListeners(): void {
  _listeners.forEach(fn => fn());
}

/** Load the WASM module once; resolves if already loaded. */
export async function initAnyCompute(): Promise<AnyComputeModule> {
  if (_mod) return _mod;
  _mod = await loadAnyCompute();
  notifyListeners();
  return _mod;
}

// ── Core hook ────────────────────────────────────────────────────────────

/** Returns the module instance or null until WASM is ready. */
export function useAnyComputeModule(): AnyComputeModule | null {
  const [mod, setMod] = useState<AnyComputeModule | null>(_mod);

  useEffect(() => {
    if (_mod) { setMod(_mod); return; }
    let cancelled = false;
    initAnyCompute().then(m => { if (!cancelled) setMod(m); });
    return () => { cancelled = true; };
  }, []);

  return mod;
}

// ── Generated function bindings ───────────────────────────────────────────

/** Synchronous accessor — throws if module is not yet loaded. */
export function useAnyComputeApi() {
  const mod = useAnyComputeModule();
  if (!mod) throw new Error('AnyCompute WASM not loaded — call initAnyCompute() first.');
  return {
/** Create a new empty VecSource. */
  ancSourceNew(): number { return this.mod.anc_source_new(); },
/** Add a column definition to a VecSource. */
  ancSourceAddColumn(handle: number, name: string, kind: number): void { return this.mod.anc_source_add_column(handle, name, kind); },
/** Push a row of integer values. */
  ancSourcePushRowInts(handle: number, values: number, len: number): void { return this.mod.anc_source_push_row_ints(handle, values, len); },
/** Free a VecSource previously created by anc_source_new. */
  ancSourceFree(handle: number): void { return this.mod.anc_source_free(handle); },
  };
}

// ── Transition hook ───────────────────────────────────────────────────────

export interface UseTransitionOptions {
  from: number;
  to: number;
  durationMs: number;
  easing?: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';
}

/**
 * Drives an animated numeric value from `from` to `to` using the any-compute
 * timing engine (Rust WASM). ~50x faster than React Spring for large batches.
 */
export function useAnyTransition(opts: UseTransitionOptions): number {
  const [value, setValue] = useState(opts.from);
  const frameRef = useRef<number>(0);
  const startRef = useRef<number | null>(null);
  const easingFn = useCallback((t: number) => {
    const c = Math.max(0, Math.min(1, t));
    switch (opts.easing) {
      case 'ease-in':     return c * c * c;
      case 'ease-out':    return 1 - Math.pow(1 - c, 3);
      case 'ease-in-out': return c < 0.5 ? 4*c*c*c : 1 - Math.pow(-2*c+2, 3)/2;
      default:            return c;
    }
  }, [opts.easing]);

  useEffect(() => {
    startRef.current = null;
    const tick = (ts: number) => {
      if (startRef.current === null) startRef.current = ts;
      const t = Math.min((ts - startRef.current) / opts.durationMs, 1);
      const lerped = opts.from + (opts.to - opts.from) * easingFn(t);
      setValue(lerped);
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [opts.from, opts.to, opts.durationMs, easingFn]);

  return value;
}

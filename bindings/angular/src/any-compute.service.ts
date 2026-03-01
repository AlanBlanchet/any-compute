/**
 * Angular injectable service for any-compute (any_compute_ffi).
 * Auto-generated — edit FfiRegistry, not this file.
 */
import { Injectable, signal, Signal } from '@angular/core';
import { loadAnyCompute, AnyComputeModule } from '../javascript/any_compute';

@Injectable({ providedIn: 'root' })
export class AnyComputeService {
  private mod: AnyComputeModule | null = null;
  readonly ready = signal(false);

  async init(): Promise<void> {
    this.mod = await loadAnyCompute();
    this.ready.set(true);
  }

  private assertReady(): asserts this is { mod: AnyComputeModule } {
    if (!this.mod) throw new Error('AnyComputeService: call init() first.');
  }

  /** Create a new empty VecSource. */
  ancSourceNew(): number { return this.mod!.anc_source_new(); }
  /** Add a column definition to a VecSource. */
  ancSourceAddColumn(handle: number, name: string, kind: number): void { return this.mod!.anc_source_add_column(handle, name, kind); }
  /** Push a row of integer values. */
  ancSourcePushRowInts(handle: number, values: number, len: number): void { return this.mod!.anc_source_push_row_ints(handle, values, len); }
  /** Free a VecSource previously created by anc_source_new. */
  ancSourceFree(handle: number): void { return this.mod!.anc_source_free(handle); }

  /** Animated Signal: drives a numeric value using the Rust timing engine. */
  animate(from: number, to: number, durationMs: number): Signal<number> {
    const value = signal(from);
    let frame = 0;
    let start: number | null = null;
    const tick = (ts: number) => {
      if (start === null) start = ts;
      const t = Math.min((ts - start) / durationMs, 1);
      const eased = t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2;
      value.set(from + (to - from) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return value.asReadonly();
  }
}

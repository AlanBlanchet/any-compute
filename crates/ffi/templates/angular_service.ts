/**
 * Angular injectable service for any-compute ({{LIB_NAME}}).
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

{{GENERATED_FNS}}
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

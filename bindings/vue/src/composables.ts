/**
 * Vue 3 composables for any-compute (any_compute_ffi).
 * Auto-generated — edit FfiRegistry, not this file.
 */
import { ref, shallowRef, onMounted, onUnmounted, watch } from 'vue';
import type { Ref } from 'vue';
import { loadAnyCompute, AnyComputeModule } from '../javascript/any_compute';

let _mod: AnyComputeModule | null = null;

export async function initAnyCompute(): Promise<AnyComputeModule> {
  if (!_mod) _mod = await loadAnyCompute();
  return _mod;
}

/** Composable: reactive access to the WASM module. */
export function useAnyCompute() {
  const ready = ref(!!_mod);
  onMounted(async () => { await initAnyCompute(); ready.value = true; });
  return {
    ready,
  /** Create a new empty VecSource. */
  ancSourceNew(): number { return _mod!.anc_source_new(); },
  /** Add a column definition to a VecSource. */
  ancSourceAddColumn(handle: number, name: string, kind: number): void { return _mod!.anc_source_add_column(handle, name, kind); },
  /** Push a row of integer values. */
  ancSourcePushRowInts(handle: number, values: number, len: number): void { return _mod!.anc_source_push_row_ints(handle, values, len); },
  /** Free a VecSource previously created by anc_source_new. */
  ancSourceFree(handle: number): void { return _mod!.anc_source_free(handle); },
  };
}

/** Composable: animated numeric value driven by any-compute timing engine. */
export function useAnyTransition(
  from: Ref<number>,
  to: Ref<number>,
  durationMs: number,
): Ref<number> {
  const value = ref(from.value);
  let frame = 0, startTs: number | null = null;

  const animate = (ts: number) => {
    if (startTs === null) startTs = ts;
    const t = Math.min((ts - startTs) / durationMs, 1);
    value.value = from.value + (to.value - from.value) * t;
    if (t < 1) frame = requestAnimationFrame(animate);
  };

  watch([from, to], () => { startTs = null; cancelAnimationFrame(frame); frame = requestAnimationFrame(animate); }, { immediate: true });
  onUnmounted(() => cancelAnimationFrame(frame));
  return value;
}

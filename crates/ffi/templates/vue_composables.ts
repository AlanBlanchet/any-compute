/**
 * Vue 3 composables for any-compute ({{LIB_NAME}}).
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
{{GENERATED_FNS}}  };
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

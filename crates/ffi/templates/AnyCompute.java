// Auto-generated Java bindings for {{LIB_NAME}}
// Uses JNI (Java 8+) or Panama FFM (Java 22+)

package com.anycompute;

import java.lang.foreign.*;
import java.lang.invoke.*;

/**
 * Native bindings to any-compute-ffi.
 * Uses the Panama Foreign Function & Memory API (Java 22+).
 */
public class AnyCompute implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIB;

    static {
        System.loadLibrary("any_compute_ffi");
        LIB = SymbolLookup.loaderLookup();
    }

{{METHOD_HANDLES}}

    private MemorySegment handle;

    public AnyCompute() {
        try {
            this.handle = (MemorySegment) ANC_SOURCE_NEW.invokeExact();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create source", e);
        }
    }

    @Override
    public void close() {
        if (handle != null) {
            try {
                ANC_SOURCE_FREE.invokeExact(handle);
            } catch (Throwable e) {
                throw new RuntimeException("Failed to free source", e);
            }
            handle = null;
        }
    }
}

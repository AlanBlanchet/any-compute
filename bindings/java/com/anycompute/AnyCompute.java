// Auto-generated Java bindings for any_compute_ffi
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

    // Create a new empty VecSource.
    private static final MethodHandle ANC_SOURCE_NEW = LINKER.downcallHandle(
        LIB.find("anc_source_new").orElseThrow(), FunctionDescriptor.of(ValueLayout.ADDRESS, ));

    // Add a column definition to a VecSource.
    private static final MethodHandle ANC_SOURCE_ADD_COLUMN = LINKER.downcallHandle(
        LIB.find("anc_source_add_column").orElseThrow(), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BYTE));

    // Push a row of integer values.
    private static final MethodHandle ANC_SOURCE_PUSH_ROW_INTS = LINKER.downcallHandle(
        LIB.find("anc_source_push_row_ints").orElseThrow(), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    // Free a VecSource previously created by anc_source_new.
    private static final MethodHandle ANC_SOURCE_FREE = LINKER.downcallHandle(
        LIB.find("anc_source_free").orElseThrow(), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

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

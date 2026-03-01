//! # any-compute-ffi
//!
//! C ABI surface for `any-compute-core`.
//!
//! This crate compiles to a `cdylib` / `staticlib` that Python, JS (via WASM),
//! and other languages can call through their FFI mechanisms.
//!
//! ## Adding a new binding
//! 1. Write the Rust function with `#[unsafe(no_mangle)] pub unsafe extern "C"`.
//! 2. Keep memory ownership rules crystal-clear: caller-allocates or callee-allocates.
//! 3. Expose an `anc_*_free` companion for every allocation this side does.

pub mod codegen;

use std::ffi::CStr;
use std::os::raw::c_char;

use any_compute_core::data::{CellValue, ColumnKind, ColumnMeta, VecSource};

/// Opaque handle to a `VecSource`.
pub type SourceHandle = *mut VecSource;

/// Create a new empty `VecSource`.
///
/// # Safety
/// Caller must eventually call [`anc_source_free`] on the returned handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anc_source_new() -> SourceHandle {
    Box::into_raw(Box::new(VecSource {
        columns: Vec::new(),
        rows: Vec::new(),
    }))
}

/// Add a column definition.
///
/// # Safety
/// `handle` must be a valid `SourceHandle`. `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anc_source_add_column(
    handle: SourceHandle,
    name: *const c_char,
    kind: u8,
) {
    let src = unsafe { &mut *handle };
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let kind = match kind {
        0 => ColumnKind::Bool,
        1 => ColumnKind::Int,
        2 => ColumnKind::Float,
        _ => ColumnKind::Text,
    };
    src.columns.push(ColumnMeta { name, kind });
}

/// Push a row of integer values (simplified — extend for mixed types).
///
/// # Safety
/// `values` must point to `len` valid `i64`s. `handle` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anc_source_push_row_ints(
    handle: SourceHandle,
    values: *const i64,
    len: usize,
) {
    let src = unsafe { &mut *handle };
    let slice = unsafe { std::slice::from_raw_parts(values, len) };
    let row: Vec<CellValue> = slice.iter().map(|&v| CellValue::Int(v)).collect();
    src.rows.push(row);
}

/// Free a `VecSource` previously created by `anc_source_new`.
///
/// # Safety
/// `handle` must be a valid, non-null pointer from `anc_source_new`.
/// Must not be called more than once for the same handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anc_source_free(handle: SourceHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

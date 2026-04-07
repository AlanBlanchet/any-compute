use serde::{Deserialize, Serialize};

/// Primitive types supported across the FFI boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FfiType {
    Void,
    Bool,
    U8,
    I32,
    I64,
    U64,
    Usize,
    F32,
    F64,
    /// Opaque pointer (`*mut T` or `*const T`).
    OpaquePtr,
    /// Null-terminated C string (`*const c_char`).
    CStr,
    /// Pointer to a typed array + length.
    Slice(SliceElementType),
}

/// Element type for slice parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceElementType {
    I64,
    F64,
    U8,
}

/// A single FFI function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiFunction {
    /// The C symbol name (e.g. `anc_source_new`).
    pub name: String,
    /// Doc comment / purpose.
    pub doc: String,
    /// Parameters in order.
    pub params: Vec<FfiParam>,
    /// Return type.
    pub ret: FfiType,
    /// Whether a matching `_free` function exists (for allocators).
    pub has_free: bool,
}

/// A single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiParam {
    pub name: String,
    pub ty: FfiType,
}

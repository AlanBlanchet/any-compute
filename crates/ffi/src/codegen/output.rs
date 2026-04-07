use super::types::*;

pub struct PythonOutput {
    pub wrapper: String,
    pub tests: String,
}

#[derive(Debug, Clone)]
pub struct JavaScriptOutput {
    pub wrapper: String,
    pub tests: String,
    pub types: String,
}

#[derive(Debug, Clone)]
pub struct JavaOutput {
    pub wrapper: String,
    pub tests: String,
}

#[derive(Debug, Clone)]
pub struct ReactOutput {
    pub hooks: String,
    pub bench: String,
    pub package_json: String,
}

#[derive(Debug, Clone)]
pub struct VueOutput {
    pub composables: String,
    pub package_json: String,
}

#[derive(Debug, Clone)]
pub struct SvelteOutput {
    pub stores: String,
    pub package_json: String,
}

#[derive(Debug, Clone)]
pub struct AngularOutput {
    pub service: String,
    pub module: String,
    pub package_json: String,
}

#[derive(Debug, Clone)]
pub struct NodeOutput {
    pub index: String,
    pub bench: String,
    pub package_json: String,
}

// ── Type mapping helpers ──────────────────────────────────────────────────

pub(super) fn ffi_type_to_python(ty: &FfiType) -> String {
    match ty {
        FfiType::Void => "None".into(),
        FfiType::Bool => "ctypes.c_bool".into(),
        FfiType::U8 => "ctypes.c_uint8".into(),
        FfiType::I32 => "ctypes.c_int32".into(),
        FfiType::I64 => "ctypes.c_int64".into(),
        FfiType::U64 => "ctypes.c_uint64".into(),
        FfiType::Usize => "ctypes.c_size_t".into(),
        FfiType::F32 => "ctypes.c_float".into(),
        FfiType::F64 => "ctypes.c_double".into(),
        FfiType::OpaquePtr => "ctypes.c_void_p".into(),
        FfiType::CStr => "ctypes.c_char_p".into(),
        FfiType::Slice(SliceElementType::I64) => "ctypes.POINTER(ctypes.c_int64)".into(),
        FfiType::Slice(SliceElementType::F64) => "ctypes.POINTER(ctypes.c_double)".into(),
        FfiType::Slice(SliceElementType::U8) => "ctypes.POINTER(ctypes.c_uint8)".into(),
    }
}

pub(super) fn ffi_type_to_ts(ty: &FfiType) -> String {
    match ty {
        FfiType::Void => "void".into(),
        FfiType::Bool => "boolean".into(),
        FfiType::U8 | FfiType::I32 | FfiType::I64 | FfiType::U64 | FfiType::Usize => {
            "number".into()
        }
        FfiType::F32 | FfiType::F64 => "number".into(),
        FfiType::OpaquePtr => "number".into(), // WASM pointers are i32
        FfiType::CStr => "string".into(),
        FfiType::Slice(_) => "number".into(), // pointer
    }
}

pub(super) fn ffi_type_to_java_layout(ty: &FfiType) -> String {
    match ty {
        FfiType::Void => "ValueLayout.ADDRESS".into(), // placeholder
        FfiType::Bool => "ValueLayout.JAVA_BOOLEAN".into(),
        FfiType::U8 => "ValueLayout.JAVA_BYTE".into(),
        FfiType::I32 => "ValueLayout.JAVA_INT".into(),
        FfiType::I64 => "ValueLayout.JAVA_LONG".into(),
        FfiType::U64 => "ValueLayout.JAVA_LONG".into(),
        FfiType::Usize => "ValueLayout.JAVA_LONG".into(),
        FfiType::F32 => "ValueLayout.JAVA_FLOAT".into(),
        FfiType::F64 => "ValueLayout.JAVA_DOUBLE".into(),
        FfiType::OpaquePtr => "ValueLayout.ADDRESS".into(),
        FfiType::CStr => "ValueLayout.ADDRESS".into(),
        FfiType::Slice(_) => "ValueLayout.ADDRESS".into(),
    }
}

/// Convert `snake_case` to `camelCase` for JS/TS bindings.
pub(super) fn to_camel(s: &str) -> String {
    let mut out = String::new();
    let mut upper_next = false;
    for (i, ch) in s.chars().enumerate() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next && i > 0 {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

// generate_all() is in mod.rs where all generate_* functions are available.

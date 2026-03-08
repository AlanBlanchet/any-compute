//! FFI code generation — auto-generate bindings for Python, JavaScript, and Java.
//!
//! This module parses the FFI surface defined in `any-compute-ffi` and generates
//! language-specific wrapper code + test scaffolding.
//!
//! ## How it works
//!
//! 1. Define FFI functions in `crates/ffi/src/lib.rs` with `#[unsafe(no_mangle)]`
//! 2. Register them in a [`FfiRegistry`] with type metadata
//! 3. Call `generate_*()` to emit wrapper code for each target language
//!
//! ## Supported targets
//!
//! | Language   | Binding style          | Test framework       |
//! |------------|------------------------|----------------------|
//! | Python     | ctypes / cffi          | pytest               |
//! | JavaScript | WASM (wasm-bindgen)    | vitest / jest        |
//! | Java       | JNI / Panama (FFM)     | JUnit 5              |

use serde::{Deserialize, Serialize};
use std::fmt::Write;

// ── FFI type model ────────────────────────────────────────────────────────

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

/// Registry of all FFI functions — the single source of truth for codegen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FfiRegistry {
    pub lib_name: String,
    pub functions: Vec<FfiFunction>,
}

impl FfiRegistry {
    pub fn new(lib_name: &str) -> Self {
        Self {
            lib_name: lib_name.to_string(),
            functions: Vec::new(),
        }
    }

    pub fn register(&mut self, func: FfiFunction) {
        self.functions.push(func);
    }

    /// Build the default registry from the any-compute-ffi surface.
    pub fn default_any_compute() -> Self {
        let mut reg = Self::new("any_compute_ffi");

        reg.register(FfiFunction {
            name: "anc_source_new".into(),
            doc: "Create a new empty VecSource.".into(),
            params: vec![],
            ret: FfiType::OpaquePtr,
            has_free: true,
        });

        reg.register(FfiFunction {
            name: "anc_source_add_column".into(),
            doc: "Add a column definition to a VecSource.".into(),
            params: vec![
                FfiParam {
                    name: "handle".into(),
                    ty: FfiType::OpaquePtr,
                },
                FfiParam {
                    name: "name".into(),
                    ty: FfiType::CStr,
                },
                FfiParam {
                    name: "kind".into(),
                    ty: FfiType::U8,
                },
            ],
            ret: FfiType::Void,
            has_free: false,
        });

        reg.register(FfiFunction {
            name: "anc_source_push_row_ints".into(),
            doc: "Push a row of integer values.".into(),
            params: vec![
                FfiParam {
                    name: "handle".into(),
                    ty: FfiType::OpaquePtr,
                },
                FfiParam {
                    name: "values".into(),
                    ty: FfiType::Slice(SliceElementType::I64),
                },
                FfiParam {
                    name: "len".into(),
                    ty: FfiType::Usize,
                },
            ],
            ret: FfiType::Void,
            has_free: false,
        });

        reg.register(FfiFunction {
            name: "anc_source_free".into(),
            doc: "Free a VecSource previously created by anc_source_new.".into(),
            params: vec![FfiParam {
                name: "handle".into(),
                ty: FfiType::OpaquePtr,
            }],
            ret: FfiType::Void,
            has_free: false,
        });

        reg
    }
}

// ── Template loading ─────────────────────────────────────────────────────

mod tpl {
    pub const PYTHON_WRAPPER: &str = include_str!("../templates/wrapper.py");
    pub const PYTHON_TESTS: &str = include_str!("../templates/test.py");
    pub const JS_WRAPPER: &str = include_str!("../templates/wrapper.js");
    pub const JS_TESTS: &str = include_str!("../templates/test.js");
    pub const TS_TYPES: &str = include_str!("../templates/types.d.ts");
    pub const JAVA_WRAPPER: &str = include_str!("../templates/AnyCompute.java");
    pub const JAVA_TESTS: &str = include_str!("../templates/AnyComputeTest.java");

    // Framework templates
    pub const REACT_HOOKS: &str = include_str!("../templates/react_hooks.ts");
    pub const REACT_BENCH: &str = include_str!("../templates/react_bench.ts");
    pub const REACT_PKG: &str = include_str!("../templates/react_package.json");
    pub const VUE_COMPOSABLES: &str = include_str!("../templates/vue_composables.ts");
    pub const VUE_PKG: &str = include_str!("../templates/vue_package.json");
    pub const SVELTE_STORES: &str = include_str!("../templates/svelte_stores.ts");
    pub const SVELTE_PKG: &str = include_str!("../templates/svelte_package.json");
    pub const ANGULAR_SERVICE: &str = include_str!("../templates/angular_service.ts");
    pub const ANGULAR_MODULE: &str = include_str!("../templates/angular_module.ts");
    pub const ANGULAR_PKG: &str = include_str!("../templates/angular_package.json");
    pub const NODE_INDEX: &str = include_str!("../templates/node_index.ts");
    pub const NODE_BENCH: &str = include_str!("../templates/node_bench.ts");
    pub const NODE_PKG: &str = include_str!("../templates/node_package.json");

    /// Replace `{{KEY}}` placeholders in a template with concrete values.
    pub fn instantiate(template: &str, vars: &[(&str, &str)]) -> String {
        let mut out = template.to_string();
        for &(key, val) in vars {
            out = out.replace(&format!("{{{{{key}}}}}"), val);
        }
        out
    }
}

// ── Code generators ──────────────────────────────────────────────────────

/// Generate Python (ctypes) wrapper + pytest tests.
pub fn generate_python(registry: &FfiRegistry) -> PythonOutput {
    // Build the data-driven function declarations
    let mut decls = String::new();
    for func in &registry.functions {
        writeln!(decls, "# {}", func.doc).unwrap();
        let argtypes: Vec<String> = func
            .params
            .iter()
            .map(|p| ffi_type_to_python(&p.ty))
            .collect();
        writeln!(
            decls,
            "_lib.{}.argtypes = [{}]",
            func.name,
            argtypes.join(", ")
        )
        .unwrap();
        writeln!(
            decls,
            "_lib.{}.restype = {}",
            func.name,
            ffi_type_to_python(&func.ret)
        )
        .unwrap();
        writeln!(decls).unwrap();
    }

    let wrapper = tpl::instantiate(
        tpl::PYTHON_WRAPPER,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("FUNCTION_DECLARATIONS", decls.trim()),
        ],
    );
    let tests = tpl::instantiate(tpl::PYTHON_TESTS, &[("LIB_NAME", &registry.lib_name)]);

    PythonOutput { wrapper, tests }
}

/// Generate JavaScript/TypeScript bindings (WASM-style).
pub fn generate_javascript(registry: &FfiRegistry) -> JavaScriptOutput {
    // Build the data-driven TypeScript interface members
    let mut ts_members = String::new();
    for func in &registry.functions {
        let ts_params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, ffi_type_to_ts(&p.ty)))
            .collect();
        writeln!(ts_members, "  /** {} */", func.doc).unwrap();
        writeln!(
            ts_members,
            "  {}({}): {};",
            func.name,
            ts_params.join(", "),
            ffi_type_to_ts(&func.ret)
        )
        .unwrap();
    }

    let wrapper = tpl::instantiate(tpl::JS_WRAPPER, &[("LIB_NAME", &registry.lib_name)]);
    let tests = tpl::instantiate(tpl::JS_TESTS, &[("LIB_NAME", &registry.lib_name)]);
    let types = tpl::instantiate(
        tpl::TS_TYPES,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("FUNCTION_TYPES", ts_members.trim_end()),
        ],
    );

    JavaScriptOutput {
        wrapper,
        tests,
        types,
    }
}

/// Generate React TypeScript hooks wrapping the WASM module.
pub fn generate_react(registry: &FfiRegistry) -> ReactOutput {
    let mut hook_fns = String::new();
    for func in &registry.functions {
        let ts_params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, ffi_type_to_ts(&p.ty)))
            .collect();
        let call_args: Vec<&str> = func.params.iter().map(|p| p.name.as_str()).collect();
        writeln!(hook_fns, "/** {} */", func.doc).unwrap();
        writeln!(
            hook_fns,
            "  {}({}): {} {{ return this.mod.{}({}); }},",
            to_camel(&func.name),
            ts_params.join(", "),
            ffi_type_to_ts(&func.ret),
            func.name,
            call_args.join(", "),
        )
        .unwrap();
    }

    let hooks = tpl::instantiate(
        tpl::REACT_HOOKS,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("GENERATED_FNS", &hook_fns),
        ],
    );

    ReactOutput {
        hooks,
        bench: tpl::REACT_BENCH.to_string(),
        package_json: tpl::REACT_PKG.to_string(),
    }
}

/// Generate Vue 3 TypeScript composables wrapping the WASM module.
pub fn generate_vue(registry: &FfiRegistry) -> VueOutput {
    let mut composable_fns = String::new();
    for func in &registry.functions {
        let ts_params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, ffi_type_to_ts(&p.ty)))
            .collect();
        let call_args: Vec<&str> = func.params.iter().map(|p| p.name.as_str()).collect();
        writeln!(composable_fns, "  /** {} */", func.doc).unwrap();
        writeln!(
            composable_fns,
            "  {}({}): {} {{ return _mod!.{}({}); }},",
            to_camel(&func.name),
            ts_params.join(", "),
            ffi_type_to_ts(&func.ret),
            func.name,
            call_args.join(", "),
        )
        .unwrap();
    }

    let composables = tpl::instantiate(
        tpl::VUE_COMPOSABLES,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("GENERATED_FNS", &composable_fns),
        ],
    );

    VueOutput {
        composables,
        package_json: tpl::VUE_PKG.to_string(),
    }
}

/// Generate Svelte stores and actions wrapping the WASM module.
pub fn generate_svelte(registry: &FfiRegistry) -> SvelteOutput {
    let mut store_fns = String::new();
    for func in &registry.functions {
        let ts_params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, ffi_type_to_ts(&p.ty)))
            .collect();
        let call_args: Vec<&str> = func.params.iter().map(|p| p.name.as_str()).collect();
        writeln!(store_fns, "  /** {} */", func.doc).unwrap();
        writeln!(
            store_fns,
            "  {}({}): {} {{ return get(mod)!.{}({}); }},",
            to_camel(&func.name),
            ts_params.join(", "),
            ffi_type_to_ts(&func.ret),
            func.name,
            call_args.join(", "),
        )
        .unwrap();
    }

    let stores = tpl::instantiate(
        tpl::SVELTE_STORES,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("GENERATED_FNS", &store_fns),
        ],
    );

    SvelteOutput {
        stores,
        package_json: tpl::SVELTE_PKG.to_string(),
    }
}

/// Generate Angular injectable service wrapping the WASM module.
pub fn generate_angular(registry: &FfiRegistry) -> AngularOutput {
    let mut service_methods = String::new();
    for func in &registry.functions {
        let ts_params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, ffi_type_to_ts(&p.ty)))
            .collect();
        let call_args: Vec<&str> = func.params.iter().map(|p| p.name.as_str()).collect();
        writeln!(service_methods, "  /** {} */", func.doc).unwrap();
        writeln!(
            service_methods,
            "  {}({}): {} {{ return this.mod!.{}({}); }}",
            to_camel(&func.name),
            ts_params.join(", "),
            ffi_type_to_ts(&func.ret),
            func.name,
            call_args.join(", "),
        )
        .unwrap();
    }

    let service = tpl::instantiate(
        tpl::ANGULAR_SERVICE,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("GENERATED_FNS", &service_methods),
        ],
    );

    AngularOutput {
        service,
        module: tpl::ANGULAR_MODULE.to_string(),
        package_json: tpl::ANGULAR_PKG.to_string(),
    }
}

/// Generate Node.js native bindings (via WASM or ffi-napi).
pub fn generate_node(registry: &FfiRegistry) -> NodeOutput {
    let mut exports = String::new();
    for func in &registry.functions {
        let ts_params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, ffi_type_to_ts(&p.ty)))
            .collect();
        let call_args: Vec<&str> = func.params.iter().map(|p| p.name.as_str()).collect();
        writeln!(exports, "/** {} */", func.doc).unwrap();
        writeln!(
            exports,
            "export function {}({}): {} {{ return _mod.{}({}); }}",
            to_camel(&func.name),
            ts_params.join(", "),
            ffi_type_to_ts(&func.ret),
            func.name,
            call_args.join(", "),
        )
        .unwrap();
    }

    let index = tpl::instantiate(
        tpl::NODE_INDEX,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("GENERATED_FNS", &exports),
        ],
    );

    NodeOutput {
        index,
        bench: tpl::NODE_BENCH.to_string(),
        package_json: tpl::NODE_PKG.to_string(),
    }
}

/// Generate Java (JNI / Panama FFM) bindings.
pub fn generate_java(registry: &FfiRegistry) -> JavaOutput {
    // Build the data-driven MethodHandle declarations
    let mut handles = String::new();
    for func in &registry.functions {
        let java_ret = ffi_type_to_java_layout(&func.ret);
        let java_params: Vec<String> = func
            .params
            .iter()
            .map(|p| ffi_type_to_java_layout(&p.ty))
            .collect();
        writeln!(handles, "    // {}", func.doc).unwrap();
        let fd = if func.ret == FfiType::Void {
            format!("FunctionDescriptor.ofVoid({})", java_params.join(", "))
        } else {
            format!(
                "FunctionDescriptor.of({}, {})",
                java_ret,
                java_params.join(", ")
            )
        };
        writeln!(
            handles,
            "    private static final MethodHandle {} = LINKER.downcallHandle(",
            func.name.to_uppercase()
        )
        .unwrap();
        writeln!(
            handles,
            "        LIB.find(\"{}\").orElseThrow(), {});",
            func.name, fd
        )
        .unwrap();
        writeln!(handles).unwrap();
    }

    let wrapper = tpl::instantiate(
        tpl::JAVA_WRAPPER,
        &[
            ("LIB_NAME", &registry.lib_name),
            ("METHOD_HANDLES", handles.trim_end()),
        ],
    );
    let tests = tpl::instantiate(tpl::JAVA_TESTS, &[("LIB_NAME", &registry.lib_name)]);

    JavaOutput { wrapper, tests }
}

// ── Output types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
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

fn ffi_type_to_python(ty: &FfiType) -> String {
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

fn ffi_type_to_ts(ty: &FfiType) -> String {
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

fn ffi_type_to_java_layout(ty: &FfiType) -> String {
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
fn to_camel(s: &str) -> String {
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

// ── Convenience: write generated code to disk ─────────────────────────────

/// Generate all bindings and write them to the given output directory.
pub fn generate_all(registry: &FfiRegistry, out_dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;

    // Python
    let py = generate_python(registry);
    let py_dir = out_dir.join("python");
    std::fs::create_dir_all(&py_dir)?;
    std::fs::write(py_dir.join("any_compute.py"), &py.wrapper)?;
    std::fs::write(py_dir.join("test_any_compute.py"), &py.tests)?;

    // JavaScript / TypeScript (WASM core)
    let js = generate_javascript(registry);
    let js_dir = out_dir.join("javascript");
    std::fs::create_dir_all(&js_dir)?;
    std::fs::write(js_dir.join("any_compute.js"), &js.wrapper)?;
    std::fs::write(js_dir.join("any_compute.d.ts"), &js.types)?;
    std::fs::write(js_dir.join("any_compute.test.js"), &js.tests)?;

    // Java
    let java = generate_java(registry);
    let java_dir = out_dir.join("java/com/anycompute");
    std::fs::create_dir_all(&java_dir)?;
    std::fs::write(java_dir.join("AnyCompute.java"), &java.wrapper)?;
    std::fs::write(java_dir.join("AnyComputeTest.java"), &java.tests)?;

    // React
    let react = generate_react(registry);
    let react_dir = out_dir.join("react/src");
    std::fs::create_dir_all(&react_dir)?;
    std::fs::write(react_dir.join("hooks.ts"), &react.hooks)?;
    std::fs::write(react_dir.join("bench.ts"), &react.bench)?;
    std::fs::write(out_dir.join("react/package.json"), &react.package_json)?;

    // Vue
    let vue = generate_vue(registry);
    let vue_dir = out_dir.join("vue/src");
    std::fs::create_dir_all(&vue_dir)?;
    std::fs::write(vue_dir.join("composables.ts"), &vue.composables)?;
    std::fs::write(out_dir.join("vue/package.json"), &vue.package_json)?;

    // Svelte
    let svelte = generate_svelte(registry);
    let svelte_dir = out_dir.join("svelte/src");
    std::fs::create_dir_all(&svelte_dir)?;
    std::fs::write(svelte_dir.join("stores.ts"), &svelte.stores)?;
    std::fs::write(out_dir.join("svelte/package.json"), &svelte.package_json)?;

    // Angular
    let angular = generate_angular(registry);
    let angular_dir = out_dir.join("angular/src");
    std::fs::create_dir_all(&angular_dir)?;
    std::fs::write(angular_dir.join("any-compute.service.ts"), &angular.service)?;
    std::fs::write(angular_dir.join("any-compute.module.ts"), &angular.module)?;
    std::fs::write(out_dir.join("angular/package.json"), &angular.package_json)?;

    // Node.js
    let node = generate_node(registry);
    let node_dir = out_dir.join("node/src");
    std::fs::create_dir_all(&node_dir)?;
    std::fs::write(node_dir.join("index.ts"), &node.index)?;
    std::fs::write(node_dir.join("bench.ts"), &node.bench)?;
    std::fs::write(out_dir.join("node/package.json"), &node.package_json)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> FfiRegistry {
        FfiRegistry::default_any_compute()
    }

    #[test]
    fn python_output_contains_ctypes() {
        let py = generate_python(&registry());
        assert!(py.wrapper.contains("ctypes.CDLL"));
        assert!(py.wrapper.contains("anc_source_new"));
        assert!(py.wrapper.contains("class VecSource"));
    }

    #[test]
    fn python_tests_contain_pytest() {
        let py = generate_python(&registry());
        assert!(py.tests.contains("def test_create_and_free"));
        assert!(py.tests.contains("def test_push_rows"));
    }

    #[test]
    fn javascript_output_contains_wasm() {
        let js = generate_javascript(&registry());
        assert!(js.wrapper.contains("WebAssembly.instantiate"));
        assert!(js.wrapper.contains("class VecSource"));
    }

    #[test]
    fn typescript_types_generated() {
        let js = generate_javascript(&registry());
        assert!(js.types.contains("interface AnyComputeModule"));
        assert!(js.types.contains("anc_source_new"));
    }

    #[test]
    fn javascript_tests_contain_vitest() {
        let js = generate_javascript(&registry());
        assert!(js.tests.contains("describe('VecSource'"));
        assert!(js.tests.contains("import { describe, it, expect }"));
    }

    #[test]
    fn java_output_contains_panama() {
        let java = generate_java(&registry());
        assert!(java.wrapper.contains("java.lang.foreign"));
        assert!(java.wrapper.contains("MethodHandle"));
        assert!(java.wrapper.contains("ANC_SOURCE_NEW"));
    }

    #[test]
    fn java_tests_contain_junit() {
        let java = generate_java(&registry());
        assert!(java.tests.contains("@Test"));
        assert!(java.tests.contains("createAndFree"));
    }

    #[test]
    fn registry_has_all_functions() {
        let reg = registry();
        assert_eq!(reg.functions.len(), 4);
        assert!(reg.functions.iter().any(|f| f.name == "anc_source_new"));
        assert!(reg.functions.iter().any(|f| f.name == "anc_source_free"));
    }

    #[test]
    fn generate_all_writes_files() {
        let reg = registry();
        let tmp = std::env::temp_dir().join("any_compute_codegen_test");
        let _ = std::fs::remove_dir_all(&tmp);
        generate_all(&reg, &tmp).unwrap();

        let expected = [
            "python/any_compute.py",
            "python/test_any_compute.py",
            "javascript/any_compute.js",
            "javascript/any_compute.d.ts",
            "javascript/any_compute.test.js",
            "java/com/anycompute/AnyCompute.java",
            "java/com/anycompute/AnyComputeTest.java",
            "react/src/hooks.ts",
            "react/src/bench.ts",
            "react/package.json",
            "vue/src/composables.ts",
            "vue/package.json",
            "svelte/src/stores.ts",
            "svelte/package.json",
            "angular/src/any-compute.service.ts",
            "angular/src/any-compute.module.ts",
            "angular/package.json",
            "node/src/index.ts",
            "node/src/bench.ts",
            "node/package.json",
        ];
        for path in &expected {
            assert!(tmp.join(path).exists(), "missing: {path}");
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}

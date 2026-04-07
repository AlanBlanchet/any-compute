//! Built-in runtime objects and functions.
//!
//! Installed into the VM's global scope at startup. Covers the minimum
//! useful subset for DOM/compute interop:
//!
//! - `console.log` — output for debugging
//! - `Math.*` — standard math functions (maps to Rust f64 methods)
//! - `JSON.stringify` / `JSON.parse` — serialization
//! - `Object`, `Array` statics
//!
//! ## Extension points
//!
//! Games, DOM bindings, and compute APIs register additional globals
//! via `Vm::define_native_fn` or by injecting a `JsObject` into `vm.globals`.

use super::JsError;
use super::value::{JsObject, JsValue};
use super::vm::Vm;

/// Install all built-in objects into the VM global scope.
pub fn install_builtins(vm: &mut Vm) {
    install_console(vm);
    install_math(vm);
    install_json(vm);
}

// ═══════════════════════════════════════════════════════════════════════════
// ── console ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

fn install_console(vm: &mut Vm) {
    let mut console = JsObject::new();
    console.set(
        "log".into(),
        JsValue::Object(JsObject::native_fn(console_log)),
    );
    console.set(
        "warn".into(),
        JsValue::Object(JsObject::native_fn(console_log)),
    );
    console.set(
        "error".into(),
        JsValue::Object(JsObject::native_fn(console_log)),
    );
    vm.define_global("console", JsValue::Object(console));
}

fn console_log(_vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let msg: Vec<String> = args.iter().map(|a| a.to_js_string()).collect();
    log::info!("[js] {}", msg.join(" "));
    Ok(JsValue::Undefined)
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Math ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Macro to generate Math.* one-arg f64 functions.
macro_rules! math_unary {
    ($($name:ident => $method:ident),* $(,)?) => {
        fn install_math(vm: &mut Vm) {
            let mut math = JsObject::new();
            // Constants
            math.set("PI".into(), JsValue::Number(std::f64::consts::PI));
            math.set("E".into(), JsValue::Number(std::f64::consts::E));
            math.set("LN2".into(), JsValue::Number(std::f64::consts::LN_2));
            math.set("LN10".into(), JsValue::Number(std::f64::consts::LN_10));
            math.set("SQRT2".into(), JsValue::Number(std::f64::consts::SQRT_2));
            math.set("Infinity".into(), JsValue::Number(f64::INFINITY));
            math.set("NaN".into(), JsValue::Number(f64::NAN));

            // Unary functions
            $(
                math.set(stringify!($name).into(), JsValue::Object(JsObject::native_fn(
                    |_vm: &mut Vm, _this: &JsValue, args: &[JsValue]| -> Result<JsValue, JsError> {
                        let x = args.first().map(|a| a.to_number()).unwrap_or(f64::NAN);
                        Ok(JsValue::Number(x.$method()))
                    }
                )));
            )*

            // Multi-arg functions
            math.set("max".into(), JsValue::Object(JsObject::native_fn(math_max)));
            math.set("min".into(), JsValue::Object(JsObject::native_fn(math_min)));
            math.set("pow".into(), JsValue::Object(JsObject::native_fn(math_pow)));
            math.set("random".into(), JsValue::Object(JsObject::native_fn(math_random)));

            vm.define_global("Math", JsValue::Object(math));
        }
    }
}

math_unary! {
    abs   => abs,
    ceil  => ceil,
    floor => floor,
    round => round,
    sqrt  => sqrt,
    cbrt  => cbrt,
    sin   => sin,
    cos   => cos,
    tan   => tan,
    asin  => asin,
    acos  => acos,
    atan  => atan,
    exp   => exp,
    log   => ln,
    log2  => log2,
    log10 => log10,
    trunc => trunc,
    sign  => signum,
}

fn math_max(_vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let result = args
        .iter()
        .map(|a| a.to_number())
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(JsValue::Number(result))
}

fn math_min(_vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let result = args
        .iter()
        .map(|a| a.to_number())
        .fold(f64::INFINITY, f64::min);
    Ok(JsValue::Number(result))
}

fn math_pow(_vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let base = args.first().map(|a| a.to_number()).unwrap_or(f64::NAN);
    let exp = args.get(1).map(|a| a.to_number()).unwrap_or(f64::NAN);
    Ok(JsValue::Number(base.powf(exp)))
}

fn math_random(_vm: &mut Vm, _this: &JsValue, _args: &[JsValue]) -> Result<JsValue, JsError> {
    // Simple pseudo-random — for deterministic tests, seed externally
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    Ok(JsValue::Number((nanos as f64 / u32::MAX as f64).fract()))
}

// ═══════════════════════════════════════════════════════════════════════════
// ── JSON ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

fn install_json(vm: &mut Vm) {
    let mut json = JsObject::new();
    json.set(
        "stringify".into(),
        JsValue::Object(JsObject::native_fn(json_stringify)),
    );
    vm.define_global("JSON", JsValue::Object(json));
}

fn json_stringify(_vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let val = args.first().unwrap_or(&JsValue::Undefined);
    Ok(JsValue::String(js_to_json(val)))
}

fn js_to_json(val: &JsValue) -> String {
    match val {
        JsValue::Undefined => "undefined".into(),
        JsValue::Null => "null".into(),
        JsValue::Boolean(b) => if *b { "true" } else { "false" }.into(),
        JsValue::Number(n) => format!("{n}"),
        JsValue::String(s) => format!("\"{s}\""),
        JsValue::Object(obj) => {
            if !obj.elements.is_empty() {
                let elems: Vec<String> = obj.elements.iter().map(js_to_json).collect();
                format!("[{}]", elems.join(","))
            } else {
                let props: Vec<String> = obj
                    .properties
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", k, js_to_json(v)))
                    .collect();
                format!("{{{}}}", props.join(","))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_constants() {
        let mut vm = Vm::new();
        install_builtins(&mut vm);
        assert_eq!(
            vm.eval("Math.PI").unwrap(),
            JsValue::Number(std::f64::consts::PI)
        );
    }

    #[test]
    fn math_abs() {
        let mut vm = Vm::new();
        install_builtins(&mut vm);
        assert_eq!(vm.eval("Math.abs(-5)").unwrap(), JsValue::Number(5.0));
    }

    #[test]
    fn math_floor_ceil() {
        let mut vm = Vm::new();
        install_builtins(&mut vm);
        assert_eq!(vm.eval("Math.floor(3.7)").unwrap(), JsValue::Number(3.0));
        assert_eq!(vm.eval("Math.ceil(3.2)").unwrap(), JsValue::Number(4.0));
    }

    #[test]
    fn math_sqrt() {
        let mut vm = Vm::new();
        install_builtins(&mut vm);
        assert_eq!(vm.eval("Math.sqrt(144)").unwrap(), JsValue::Number(12.0));
    }

    #[test]
    fn json_stringify_number() {
        let mut vm = Vm::new();
        install_builtins(&mut vm);
        assert_eq!(
            vm.eval("JSON.stringify(42)").unwrap(),
            JsValue::String("42".into())
        );
    }
}

//! JavaScript value types — the runtime representation of all JS values.
//!
//! ## Value encoding
//!
//! Uses a tagged enum for simplicity. A future NaN-boxing optimization can
//! replace this representation without changing the public API.
//!
//! ## Heap objects
//!
//! Complex values (objects, arrays, functions) are heap-allocated via
//! reference-counted handles. The VM's garbage collector can trace roots
//! through `JsValue::Object` variants.

use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// ── JsValue ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A JavaScript value — the universal currency of the JS engine.
///
/// Follows ECMAScript semantics: `undefined`, `null`, `boolean`, `number`
/// (f64), `string`, and `object` (which covers arrays, functions, etc.).
#[derive(Debug, Clone)]
pub enum JsValue {
    Undefined,
    Null,
    Boolean(bool),
    /// IEEE 754 double — matches JS `number` semantics exactly.
    Number(f64),
    String(String),
    /// Heap-allocated object (properties + optional internal slots).
    Object(JsObject),
}

impl PartialEq for JsValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Undefined, Self::Undefined) => true,
            (Self::Null, Self::Null) => true,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            _ => false,
        }
    }
}

impl JsValue {
    // ── Type coercion (ECMAScript abstract operations) ───────────────

    /// `ToBoolean()` — falsy values: undefined, null, false, 0, NaN, "".
    pub fn to_boolean(&self) -> bool {
        match self {
            Self::Undefined | Self::Null => false,
            Self::Boolean(b) => *b,
            Self::Number(n) => *n != 0.0 && !n.is_nan(),
            Self::String(s) => !s.is_empty(),
            Self::Object(_) => true,
        }
    }

    /// `ToNumber()` — coerce to f64.
    pub fn to_number(&self) -> f64 {
        match self {
            Self::Undefined => f64::NAN,
            Self::Null => 0.0,
            Self::Boolean(true) => 1.0,
            Self::Boolean(false) => 0.0,
            Self::Number(n) => *n,
            Self::String(s) => s.parse::<f64>().unwrap_or(f64::NAN),
            Self::Object(_) => f64::NAN,
        }
    }

    /// `ToString()` — coerce to string representation.
    pub fn to_js_string(&self) -> String {
        match self {
            Self::Undefined => "undefined".into(),
            Self::Null => "null".into(),
            Self::Boolean(b) => if *b { "true" } else { "false" }.into(),
            Self::Number(n) => {
                if n.is_nan() {
                    "NaN".into()
                } else if n.is_infinite() {
                    if n.is_sign_positive() {
                        "Infinity"
                    } else {
                        "-Infinity"
                    }
                    .into()
                } else if *n == 0.0 {
                    "0".into()
                } else {
                    format!("{n}")
                }
            }
            Self::String(s) => s.clone(),
            Self::Object(_) => "[object Object]".into(),
        }
    }

    /// `typeof` operator result.
    pub fn type_of(&self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::Null => "object", // JS quirk
            Self::Boolean(_) => "boolean",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Object(obj) if obj.callable.is_some() => "function",
            Self::Object(_) => "object",
        }
    }

    /// True for `undefined` or `null`.
    pub fn is_nullish(&self) -> bool {
        matches!(self, Self::Undefined | Self::Null)
    }
}

impl fmt::Display for JsValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_js_string())
    }
}

impl From<f64> for JsValue {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

impl From<i64> for JsValue {
    fn from(n: i64) -> Self {
        Self::Number(n as f64)
    }
}

impl From<bool> for JsValue {
    fn from(b: bool) -> Self {
        Self::Boolean(b)
    }
}

impl From<String> for JsValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for JsValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── JsObject ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Native function signature — receives `this` + arguments, returns a value.
pub type NativeFn = fn(&mut super::vm::Vm, &JsValue, &[JsValue]) -> Result<JsValue, super::JsError>;

/// A JavaScript object — property map + optional internal slots.
///
/// Arrays, functions, and all user objects share this representation.
/// The `callable` slot distinguishes functions from plain objects.
#[derive(Debug, Clone)]
pub struct JsObject {
    /// Named properties (string keys for now; Symbols come later).
    pub properties: HashMap<String, JsValue>,
    /// If this object is callable, this holds the function implementation.
    pub callable: Option<Callable>,
    /// Prototype link (`__proto__`).
    pub prototype: Option<Box<JsObject>>,
    /// Array elements (if this is an array-like object).
    pub elements: Vec<JsValue>,
}

impl JsObject {
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
            callable: None,
            prototype: None,
            elements: Vec::new(),
        }
    }

    /// Create a function object wrapping a native Rust function.
    pub fn native_fn(f: NativeFn) -> Self {
        Self {
            properties: HashMap::new(),
            callable: Some(Callable::Native(f)),
            prototype: None,
            elements: Vec::new(),
        }
    }

    /// Create an array object with initial elements.
    pub fn array(elements: Vec<JsValue>) -> Self {
        let mut obj = Self::new();
        obj.properties
            .insert("length".into(), JsValue::Number(elements.len() as f64));
        obj.elements = elements;
        obj
    }

    /// Get a property, walking the prototype chain.
    pub fn get(&self, key: &str) -> JsValue {
        if let Some(val) = self.properties.get(key) {
            return val.clone();
        }
        if let Some(ref proto) = self.prototype {
            return proto.get(key);
        }
        JsValue::Undefined
    }

    /// Set a property (always on the own object, not prototype).
    pub fn set(&mut self, key: String, val: JsValue) {
        self.properties.insert(key, val);
    }
}

impl Default for JsObject {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Callable ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// How a function object is invoked.
#[derive(Clone)]
pub enum Callable {
    /// Native Rust function.
    Native(NativeFn),
    /// Bytecode function — index into the VM's function table.
    Bytecode {
        /// Index into `Vm::functions`.
        func_id: usize,
        /// Captured upvalues for closures.
        upvalues: Vec<JsValue>,
    },
}

impl fmt::Debug for Callable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(_) => write!(f, "Callable::Native(fn)"),
            Self::Bytecode { func_id, .. } => write!(f, "Callable::Bytecode({func_id})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_truthiness() {
        assert!(!JsValue::Undefined.to_boolean());
        assert!(!JsValue::Null.to_boolean());
        assert!(!JsValue::Boolean(false).to_boolean());
        assert!(!JsValue::Number(0.0).to_boolean());
        assert!(!JsValue::Number(f64::NAN).to_boolean());
        assert!(!JsValue::String("".into()).to_boolean());
        assert!(JsValue::Boolean(true).to_boolean());
        assert!(JsValue::Number(1.0).to_boolean());
        assert!(JsValue::String("x".into()).to_boolean());
    }

    #[test]
    fn value_to_number() {
        assert_eq!(JsValue::Null.to_number(), 0.0);
        assert_eq!(JsValue::Boolean(true).to_number(), 1.0);
        assert_eq!(JsValue::String("42".into()).to_number(), 42.0);
        assert!(JsValue::Undefined.to_number().is_nan());
    }

    #[test]
    fn value_type_of() {
        assert_eq!(JsValue::Undefined.type_of(), "undefined");
        assert_eq!(JsValue::Null.type_of(), "object");
        assert_eq!(JsValue::Number(1.0).type_of(), "number");
    }

    #[test]
    fn object_property_access() {
        let mut obj = JsObject::new();
        obj.set("x".into(), JsValue::Number(42.0));
        assert_eq!(obj.get("x"), JsValue::Number(42.0));
        assert_eq!(obj.get("y"), JsValue::Undefined);
    }

    #[test]
    fn from_impls() {
        assert_eq!(JsValue::from(3.14), JsValue::Number(3.14));
        assert_eq!(JsValue::from(true), JsValue::Boolean(true));
        assert_eq!(JsValue::from("hello"), JsValue::String("hello".into()));
    }
}

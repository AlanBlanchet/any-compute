//! # any-compute-js
//!
//! A V8-like JavaScript engine optimized for the any-compute ecosystem.
//!
//! ## Architecture
//!
//! ```text
//! Source code ─→ Lexer ─→ Tokens ─→ Parser ─→ AST ─→ Compiler ─→ Bytecode ─→ VM ─→ Result
//!                                                                              ↑
//!                                                                          Runtime
//!                                                                     (built-in objects,
//!                                                                      DOM bridge, etc.)
//! ```
//!
//! ## Design principles
//!
//! - **Reuse core primitives** — `V<N>`, `Buffer`, `Device` for compute-heavy operations
//! - **Register-based VM** — fewer instructions than stack-based, better for JIT later
//! - **Inline caching** — property access shapes for hot paths
//! - **Compatible, not conformant** — we implement the JS features useful for our lib's
//!   DOM/canvas/compute APIs, not the full ECMAScript spec
//!
//! ## Modules
//!
//! | Module       | Purpose                                                     |
//! |-------------|-------------------------------------------------------------|
//! | [`value`]    | JS value types (`JsValue`, `JsObject`, `JsString`, heap)    |
//! | [`lexer`]    | Source → token stream (keywords, operators, literals)        |
//! | [`parser`]   | Token stream → AST (expressions, statements, declarations)  |
//! | [`bytecode`] | AST → bytecode IR (register-addressed instructions)         |
//! | [`vm`]       | Bytecode interpreter (register file, call stack, GC roots)  |
//! | [`runtime`]  | Built-in objects (Object, Array, Math, console, DOM bridge) |

pub mod bytecode;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod value;
pub mod vm;

pub use value::JsValue;
pub use vm::Vm;

/// Create a new JS VM with the standard runtime loaded.
pub fn create_vm() -> Vm {
    let mut vm = Vm::new();
    runtime::install_builtins(&mut vm);
    vm
}

/// Evaluate a JavaScript source string and return the result.
pub fn eval(source: &str) -> Result<JsValue, JsError> {
    let mut vm = create_vm();
    vm.eval(source)
}

/// JavaScript execution error.
#[derive(Debug, Clone)]
pub enum JsError {
    /// Syntax error during lexing or parsing.
    Syntax(String),
    /// Runtime error (TypeError, ReferenceError, etc.).
    Runtime(String),
    /// Internal VM error (should not happen in normal usage).
    Internal(String),
}

impl std::fmt::Display for JsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Syntax(msg) => write!(f, "SyntaxError: {msg}"),
            Self::Runtime(msg) => write!(f, "RuntimeError: {msg}"),
            Self::Internal(msg) => write!(f, "InternalError: {msg}"),
        }
    }
}

impl std::error::Error for JsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_simple_arithmetic() {
        let result = eval("1 + 2").unwrap();
        assert_eq!(result, JsValue::Number(3.0));
    }

    #[test]
    fn eval_string_literal() {
        let result = eval("'hello'").unwrap();
        assert_eq!(result, JsValue::String("hello".into()));
    }

    #[test]
    fn eval_variable_binding() {
        let result = eval("let x = 10; x * 2").unwrap();
        assert_eq!(result, JsValue::Number(20.0));
    }
}

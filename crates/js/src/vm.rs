//! Register-based virtual machine — executes bytecode chunks.
//!
//! ## Register file
//!
//! Each call frame owns a slice of the global register file. Registers
//! are `JsValue`s — no unboxing (that's a future optimization).
//!
//! ## Scope chain
//!
//! Variables live in a chain of `Scope` hashmaps. `let`/`const` create
//! bindings in the innermost scope; `var` in the function scope.

use std::any::Any;
use std::collections::HashMap;

use super::JsError;
use super::bytecode::{Chunk, Constant, Op};
use super::value::{Callable, JsObject, JsValue};

// ═══════════════════════════════════════════════════════════════════════════
// ── Scope ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A variable scope — one level in the scope chain.
#[derive(Debug, Clone)]
struct Scope {
    vars: HashMap<String, JsValue>,
}

impl Scope {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── VM ──────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// The JavaScript virtual machine.
pub struct Vm {
    /// Global scope (built-in objects, user globals).
    pub globals: HashMap<String, JsValue>,
    /// Scope chain for the current execution context.
    scopes: Vec<Scope>,
    /// Host-provided data accessible from native functions via `Vm::user_data()`.
    /// Enables native functions (plain `fn` pointers) to interact with
    /// host-side state (DOM tree, canvas, etc.) without closures.
    host: Option<Box<dyn Any + Send>>,
}

impl Vm {
    pub fn new() -> Self {
        Self {
            globals: HashMap::new(),
            scopes: vec![Scope::new()],
            host: None,
        }
    }

    /// Attach host data that native functions can access via `user_data()`.
    pub fn set_host<T: 'static + Send>(&mut self, data: T) {
        self.host = Some(Box::new(data));
    }

    /// Downcast-ref to the host data.
    pub fn host<T: 'static>(&self) -> Option<&T> {
        self.host.as_ref()?.downcast_ref()
    }

    /// Downcast-mut to the host data.
    pub fn host_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.host.as_mut()?.downcast_mut()
    }

    /// Define a global variable.
    pub fn define_global(&mut self, name: impl Into<String>, value: JsValue) {
        self.globals.insert(name.into(), value);
    }

    /// Define a native function in the global scope.
    pub fn define_native_fn(&mut self, name: impl Into<String>, f: super::value::NativeFn) {
        let name = name.into();
        self.globals
            .insert(name, JsValue::Object(JsObject::native_fn(f)));
    }

    /// Evaluate a source string (lex → parse → compile → execute).
    pub fn eval(&mut self, source: &str) -> Result<JsValue, JsError> {
        let stmts = super::parser::parse(source)?;
        let chunk = super::bytecode::Compiler::compile_program(&stmts)?;
        self.execute(&chunk)
    }

    /// Execute a compiled chunk and return the result.
    pub fn execute(&mut self, chunk: &Chunk) -> Result<JsValue, JsError> {
        let mut regs = vec![JsValue::Undefined; chunk.reg_count as usize];
        let mut pc = 0usize;

        while pc < chunk.ops.len() {
            match &chunk.ops[pc] {
                Op::LoadConst { dst, idx } => {
                    regs[*dst as usize] = match &chunk.constants[*idx as usize] {
                        Constant::Number(n) => JsValue::Number(*n),
                        Constant::String(s) => JsValue::String(s.clone()),
                    };
                }
                Op::LoadUndefined { dst } => {
                    regs[*dst as usize] = JsValue::Undefined;
                }
                Op::LoadNull { dst } => {
                    regs[*dst as usize] = JsValue::Null;
                }
                Op::LoadBool { dst, val } => {
                    regs[*dst as usize] = JsValue::Boolean(*val);
                }
                Op::Move { dst, src } => {
                    regs[*dst as usize] = regs[*src as usize].clone();
                }

                // ── Arithmetic ──────────────────────────────────
                Op::Add { dst, a, b } => {
                    let av = &regs[*a as usize];
                    let bv = &regs[*b as usize];
                    regs[*dst as usize] = match (av, bv) {
                        (JsValue::String(s1), _) => {
                            JsValue::String(format!("{s1}{}", bv.to_js_string()))
                        }
                        (_, JsValue::String(s2)) => {
                            JsValue::String(format!("{}{s2}", av.to_js_string()))
                        }
                        _ => JsValue::Number(av.to_number() + bv.to_number()),
                    };
                }
                Op::Sub { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        regs[*a as usize].to_number() - regs[*b as usize].to_number(),
                    );
                }
                Op::Mul { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        regs[*a as usize].to_number() * regs[*b as usize].to_number(),
                    );
                }
                Op::Div { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        regs[*a as usize].to_number() / regs[*b as usize].to_number(),
                    );
                }
                Op::Mod { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        regs[*a as usize].to_number() % regs[*b as usize].to_number(),
                    );
                }
                Op::Exp { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        regs[*a as usize]
                            .to_number()
                            .powf(regs[*b as usize].to_number()),
                    );
                }
                Op::Neg { dst, src } => {
                    regs[*dst as usize] = JsValue::Number(-regs[*src as usize].to_number());
                }

                // ── Bitwise ─────────────────────────────────────
                Op::BitAnd { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        ((regs[*a as usize].to_number() as i32)
                            & (regs[*b as usize].to_number() as i32))
                            as f64,
                    );
                }
                Op::BitOr { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        ((regs[*a as usize].to_number() as i32)
                            | (regs[*b as usize].to_number() as i32))
                            as f64,
                    );
                }
                Op::BitXor { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        ((regs[*a as usize].to_number() as i32)
                            ^ (regs[*b as usize].to_number() as i32))
                            as f64,
                    );
                }
                Op::Shl { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        ((regs[*a as usize].to_number() as i32)
                            << (regs[*b as usize].to_number() as u32 & 31))
                            as f64,
                    );
                }
                Op::Shr { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        ((regs[*a as usize].to_number() as i32)
                            >> (regs[*b as usize].to_number() as u32 & 31))
                            as f64,
                    );
                }
                Op::UShr { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Number(
                        ((regs[*a as usize].to_number() as u32)
                            >> (regs[*b as usize].to_number() as u32 & 31))
                            as f64,
                    );
                }
                Op::BitNot { dst, src } => {
                    regs[*dst as usize] =
                        JsValue::Number((!(regs[*src as usize].to_number() as i32)) as f64);
                }

                // ── Comparison ──────────────────────────────────
                Op::Eq { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Boolean(regs[*a as usize] == regs[*b as usize]);
                }
                Op::StrictEq { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Boolean(regs[*a as usize] == regs[*b as usize]);
                }
                Op::Lt { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Boolean(
                        regs[*a as usize].to_number() < regs[*b as usize].to_number(),
                    );
                }
                Op::LtEq { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Boolean(
                        regs[*a as usize].to_number() <= regs[*b as usize].to_number(),
                    );
                }
                Op::Gt { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Boolean(
                        regs[*a as usize].to_number() > regs[*b as usize].to_number(),
                    );
                }
                Op::GtEq { dst, a, b } => {
                    regs[*dst as usize] = JsValue::Boolean(
                        regs[*a as usize].to_number() >= regs[*b as usize].to_number(),
                    );
                }
                Op::Not { dst, src } => {
                    regs[*dst as usize] = JsValue::Boolean(!regs[*src as usize].to_boolean());
                }

                // ── Variables ───────────────────────────────────
                Op::GetVar { dst, name_idx } => {
                    let name = &chunk.names[*name_idx as usize];
                    let val = self.get_var(name);
                    regs[*dst as usize] = val;
                }
                Op::SetVar { name_idx, src } => {
                    let name = chunk.names[*name_idx as usize].clone();
                    let val = regs[*src as usize].clone();
                    self.set_var(name, val);
                }
                Op::DeclVar { name_idx, src } => {
                    let name = chunk.names[*name_idx as usize].clone();
                    let val = regs[*src as usize].clone();
                    self.declare_var(name, val);
                }

                // ── Properties ──────────────────────────────────
                Op::GetProp { dst, obj, key_idx } => {
                    let key = &chunk.names[*key_idx as usize];
                    regs[*dst as usize] = match &regs[*obj as usize] {
                        JsValue::Object(o) => o.get(key),
                        JsValue::String(s) if key == "length" => JsValue::Number(s.len() as f64),
                        _ => JsValue::Undefined,
                    };
                }
                Op::SetProp {
                    obj,
                    key_idx,
                    value,
                } => {
                    let key = chunk.names[*key_idx as usize].clone();
                    let val = regs[*value as usize].clone();
                    if let JsValue::Object(ref mut o) = regs[*obj as usize] {
                        o.set(key, val);
                    }
                }
                Op::GetIndex { dst, obj, index } => {
                    let idx = regs[*index as usize].to_number() as usize;
                    regs[*dst as usize] = match &regs[*obj as usize] {
                        JsValue::Object(o) => {
                            o.elements.get(idx).cloned().unwrap_or(JsValue::Undefined)
                        }
                        JsValue::String(s) => s
                            .chars()
                            .nth(idx)
                            .map(|c| JsValue::String(c.to_string()))
                            .unwrap_or(JsValue::Undefined),
                        _ => JsValue::Undefined,
                    };
                }
                Op::SetIndex { obj, index, value } => {
                    let idx = regs[*index as usize].to_number() as usize;
                    let val = regs[*value as usize].clone();
                    if let JsValue::Object(ref mut o) = regs[*obj as usize] {
                        if idx >= o.elements.len() {
                            o.elements.resize(idx + 1, JsValue::Undefined);
                        }
                        o.elements[idx] = val;
                    }
                }

                // ── Control flow ────────────────────────────────
                Op::Jump { offset } => {
                    pc = (pc as i32 + offset) as usize;
                    continue;
                }
                Op::JumpIfFalse { src, offset } => {
                    if !regs[*src as usize].to_boolean() {
                        pc = (pc as i32 + offset) as usize;
                        continue;
                    }
                }
                Op::JumpIfTrue { src, offset } => {
                    if regs[*src as usize].to_boolean() {
                        pc = (pc as i32 + offset) as usize;
                        continue;
                    }
                }

                // ── Functions ───────────────────────────────────
                Op::Call {
                    dst,
                    callee,
                    args_start,
                    argc,
                } => {
                    let callee_val = regs[*callee as usize].clone();
                    let args: Vec<JsValue> = (0..*argc)
                        .map(|i| regs[(*args_start + i) as usize].clone())
                        .collect();

                    regs[*dst as usize] = match callee_val {
                        JsValue::Object(ref obj) => match &obj.callable {
                            Some(Callable::Native(f)) => f(self, &JsValue::Undefined, &args)?,
                            _ => JsValue::Undefined,
                        },
                        _ => {
                            return Err(JsError::Runtime(format!(
                                "{} is not a function",
                                callee_val.to_js_string()
                            )));
                        }
                    };
                }
                Op::CallMethod {
                    dst,
                    obj,
                    method_idx,
                    args_start,
                    argc,
                } => {
                    let receiver = regs[*obj as usize].clone();
                    let method_name = &chunk.names[*method_idx as usize];
                    let method_val = match &receiver {
                        JsValue::Object(o) => o.get(method_name),
                        JsValue::String(s) if method_name == "length" => {
                            JsValue::Number(s.len() as f64)
                        }
                        _ => JsValue::Undefined,
                    };
                    let args: Vec<JsValue> = (0..*argc)
                        .map(|i| regs[(*args_start + i) as usize].clone())
                        .collect();

                    regs[*dst as usize] = match method_val {
                        JsValue::Object(ref obj) => match &obj.callable {
                            Some(Callable::Native(f)) => f(self, &receiver, &args)?,
                            _ => JsValue::Undefined,
                        },
                        _ => {
                            return Err(JsError::Runtime(format!(
                                "{} is not a function",
                                method_name
                            )));
                        }
                    };
                }
                Op::Return { src } => {
                    return Ok(regs[*src as usize].clone());
                }

                // ── Construction ────────────────────────────────
                Op::NewObject { dst } => {
                    regs[*dst as usize] = JsValue::Object(JsObject::new());
                }
                Op::NewArray { dst, start, count } => {
                    let elems: Vec<JsValue> = (0..*count)
                        .map(|i| regs[(*start + i) as usize].clone())
                        .collect();
                    regs[*dst as usize] = JsValue::Object(JsObject::array(elems));
                }

                // ── Special ─────────────────────────────────────
                Op::Typeof { dst, src } => {
                    regs[*dst as usize] = JsValue::String(regs[*src as usize].type_of().into());
                }
            }

            pc += 1;
        }

        // Implicit return undefined
        Ok(JsValue::Undefined)
    }

    // ── Scope chain operations ─────────────────────────────────────

    fn get_var(&self, name: &str) -> JsValue {
        // Walk scope chain from innermost to outermost
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.vars.get(name) {
                return val.clone();
            }
        }
        // Fall back to globals
        self.globals
            .get(name)
            .cloned()
            .unwrap_or(JsValue::Undefined)
    }

    fn set_var(&mut self, name: String, val: JsValue) {
        // Walk scope chain from innermost to outermost
        for scope in self.scopes.iter_mut().rev() {
            if scope.vars.contains_key(&name) {
                scope.vars.insert(name, val);
                return;
            }
        }
        // Not found — set as global
        self.globals.insert(name, val);
    }

    fn declare_var(&mut self, name: String, val: JsValue) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.vars.insert(name, val);
        } else {
            self.globals.insert(name, val);
        }
    }

    /// Push a new scope (for blocks, functions).
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Pop the innermost scope.
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_simple_arithmetic() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("1 + 2").unwrap(), JsValue::Number(3.0));
        assert_eq!(vm.eval("10 - 3").unwrap(), JsValue::Number(7.0));
        assert_eq!(vm.eval("4 * 5").unwrap(), JsValue::Number(20.0));
        assert_eq!(vm.eval("10 / 3").unwrap(), JsValue::Number(10.0 / 3.0));
    }

    #[test]
    fn vm_string_concat() {
        let mut vm = Vm::new();
        assert_eq!(
            vm.eval("'hello' + ' ' + 'world'").unwrap(),
            JsValue::String("hello world".into())
        );
    }

    #[test]
    fn vm_variable_binding() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("let x = 10; x * 2").unwrap(), JsValue::Number(20.0));
    }

    #[test]
    fn vm_comparison() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("3 < 5").unwrap(), JsValue::Boolean(true));
        assert_eq!(vm.eval("5 < 3").unwrap(), JsValue::Boolean(false));
    }

    #[test]
    fn vm_typeof() {
        let mut vm = Vm::new();
        assert_eq!(
            vm.eval("typeof 42").unwrap(),
            JsValue::String("number".into())
        );
        assert_eq!(
            vm.eval("typeof 'hi'").unwrap(),
            JsValue::String("string".into())
        );
    }

    #[test]
    fn vm_boolean_not() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("!true").unwrap(), JsValue::Boolean(false));
        assert_eq!(vm.eval("!false").unwrap(), JsValue::Boolean(true));
        assert_eq!(vm.eval("!0").unwrap(), JsValue::Boolean(true));
    }

    #[test]
    fn vm_negation() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("-5").unwrap(), JsValue::Number(-5.0));
    }

    #[test]
    fn vm_native_function() {
        let mut vm = Vm::new();
        vm.define_native_fn("double", |_vm, _this, args| {
            Ok(JsValue::Number(args[0].to_number() * 2.0))
        });
        assert_eq!(vm.eval("double(21)").unwrap(), JsValue::Number(42.0));
    }

    #[test]
    fn vm_undefined_var() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("x").unwrap(), JsValue::Undefined);
    }

    #[test]
    fn vm_power_operator() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("2 ** 10").unwrap(), JsValue::Number(1024.0));
    }

    #[test]
    fn vm_modulo() {
        let mut vm = Vm::new();
        assert_eq!(vm.eval("10 % 3").unwrap(), JsValue::Number(1.0));
    }
}

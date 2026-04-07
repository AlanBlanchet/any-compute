//! Bytecode IR — AST → register-addressed instructions.
//!
//! ## Register machine
//!
//! Each function frame has a fixed set of virtual registers (r0, r1, ...).
//! The compiler assigns registers via a simple linear scan. Instructions
//! are compact tagged unions:
//!
//! ```text
//! LoadConst r0, #42      // r0 = 42
//! Add       r2, r0, r1   // r2 = r0 + r1
//! Call      r3, r0, 2    // r3 = r0(arg0, arg1) — 2 args starting after r0
//! Return    r3            // return r3
//! ```
//!
//! This is a stepping stone to JIT — the register form maps naturally
//! to machine registers when we add native codegen later.

use super::JsError;
use super::parser::{BinOp, Expr, Stmt, UnaryOp};


mod op;
pub use op::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Function chunk ──────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A compiled function — bytecode + metadata.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub name: String,
    pub params: Vec<String>,
    pub ops: Vec<Op>,
    /// Constant pool (numbers and strings interned here).
    pub constants: Vec<Constant>,
    /// Name pool (variable/property names).
    pub names: Vec<String>,
    /// How many registers this function needs.
    pub reg_count: u16,
}

/// Interned constant value.
#[derive(Debug, Clone)]
pub enum Constant {
    Number(f64),
    String(String),
}

impl Chunk {
    pub fn new(name: impl Into<String>, params: Vec<String>) -> Self {
        Self {
            name: name.into(),
            params,
            ops: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            reg_count: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Compiler ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// AST → bytecode compiler.
pub struct Compiler {
    chunk: Chunk,
    next_reg: u16,
}

impl Compiler {
    pub fn new(name: &str, params: Vec<String>) -> Self {
        Self {
            chunk: Chunk::new(name, params),
            next_reg: 0,
        }
    }

    fn alloc_reg(&mut self) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        if self.next_reg > self.chunk.reg_count {
            self.chunk.reg_count = self.next_reg;
        }
        r
    }

    fn add_constant(&mut self, c: Constant) -> u16 {
        // Dedup check
        for (i, existing) in self.chunk.constants.iter().enumerate() {
            match (existing, &c) {
                (Constant::Number(a), Constant::Number(b)) if a == b => return i as u16,
                (Constant::String(a), Constant::String(b)) if a == b => return i as u16,
                _ => {}
            }
        }
        let idx = self.chunk.constants.len() as u16;
        self.chunk.constants.push(c);
        idx
    }

    fn add_name(&mut self, name: &str) -> u16 {
        for (i, n) in self.chunk.names.iter().enumerate() {
            if n == name {
                return i as u16;
            }
        }
        let idx = self.chunk.names.len() as u16;
        self.chunk.names.push(name.to_string());
        idx
    }

    fn emit(&mut self, op: Op) -> usize {
        let idx = self.chunk.ops.len();
        self.chunk.ops.push(op);
        idx
    }

    /// Compile a full program into a top-level chunk.
    pub fn compile_program(stmts: &[Stmt]) -> Result<Chunk, JsError> {
        let mut c = Self::new("<main>", vec![]);
        let mut last_reg = None;
        for stmt in stmts {
            last_reg = Some(c.compile_stmt(stmt)?);
        }
        // Return the last expression's value
        if let Some(r) = last_reg {
            c.emit(Op::Return { src: r });
        } else {
            let r = c.alloc_reg();
            c.emit(Op::LoadUndefined { dst: r });
            c.emit(Op::Return { src: r });
        }
        Ok(c.chunk)
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<Reg, JsError> {
        match stmt {
            Stmt::Expr(expr) => self.compile_expr(expr),
            Stmt::VarDecl {
                kind: _,
                name,
                init,
            } => {
                let r = if let Some(init) = init {
                    self.compile_expr(init)?
                } else {
                    let r = self.alloc_reg();
                    self.emit(Op::LoadUndefined { dst: r });
                    r
                };
                let name_idx = self.add_name(name);
                self.emit(Op::DeclVar { name_idx, src: r });
                Ok(r)
            }
            Stmt::Block(stmts) => {
                let mut last = self.alloc_reg();
                self.emit(Op::LoadUndefined { dst: last });
                for s in stmts {
                    last = self.compile_stmt(s)?;
                }
                Ok(last)
            }
            Stmt::Return(expr) => {
                let r = if let Some(e) = expr {
                    self.compile_expr(e)?
                } else {
                    let r = self.alloc_reg();
                    self.emit(Op::LoadUndefined { dst: r });
                    r
                };
                self.emit(Op::Return { src: r });
                Ok(r)
            }
            Stmt::If {
                condition,
                consequent,
                alternate,
            } => {
                let cond = self.compile_expr(condition)?;
                let jump_false = self.emit(Op::JumpIfFalse {
                    src: cond,
                    offset: 0,
                });
                let then_reg = self.compile_stmt(consequent)?;
                let result = self.alloc_reg();
                self.emit(Op::Move {
                    dst: result,
                    src: then_reg,
                });

                if let Some(alt) = alternate {
                    let jump_end = self.emit(Op::Jump { offset: 0 });
                    let else_start = self.chunk.ops.len() as i32;
                    self.chunk.ops[jump_false] = Op::JumpIfFalse {
                        src: cond,
                        offset: else_start - jump_false as i32,
                    };
                    let else_reg = self.compile_stmt(alt)?;
                    self.emit(Op::Move {
                        dst: result,
                        src: else_reg,
                    });
                    let end = self.chunk.ops.len() as i32;
                    self.chunk.ops[jump_end] = Op::Jump {
                        offset: end - jump_end as i32,
                    };
                } else {
                    let end = self.chunk.ops.len() as i32;
                    self.chunk.ops[jump_false] = Op::JumpIfFalse {
                        src: cond,
                        offset: end - jump_false as i32,
                    };
                }
                Ok(result)
            }
            Stmt::While { condition, body } => {
                let loop_start = self.chunk.ops.len() as i32;
                let cond = self.compile_expr(condition)?;
                let jump_false = self.emit(Op::JumpIfFalse {
                    src: cond,
                    offset: 0,
                });
                self.compile_stmt(body)?;
                let back = loop_start - self.chunk.ops.len() as i32;
                self.emit(Op::Jump { offset: back });
                let end = self.chunk.ops.len() as i32;
                self.chunk.ops[jump_false] = Op::JumpIfFalse {
                    src: cond,
                    offset: end - jump_false as i32,
                };
                let r = self.alloc_reg();
                self.emit(Op::LoadUndefined { dst: r });
                Ok(r)
            }
            Stmt::FunctionDecl { name, params, body } => {
                let mut sub = Compiler::new(name, params.clone());
                let mut last = sub.alloc_reg();
                sub.emit(Op::LoadUndefined { dst: last });
                for s in body {
                    last = sub.compile_stmt(s)?;
                }
                sub.emit(Op::Return { src: last });
                // Store the compiled function as a constant
                let _sub_chunk = sub.chunk;
                // For now, functions are stored by name in the runtime
                let r = self.alloc_reg();
                self.emit(Op::LoadUndefined { dst: r });
                let name_idx = self.add_name(name);
                self.emit(Op::DeclVar { name_idx, src: r });
                Ok(r)
            }
            _ => {
                let r = self.alloc_reg();
                self.emit(Op::LoadUndefined { dst: r });
                Ok(r)
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<Reg, JsError> {
        match expr {
            Expr::Number(n) => {
                let dst = self.alloc_reg();
                let idx = self.add_constant(Constant::Number(*n));
                self.emit(Op::LoadConst { dst, idx });
                Ok(dst)
            }
            Expr::String(s) => {
                let dst = self.alloc_reg();
                let idx = self.add_constant(Constant::String(s.clone()));
                self.emit(Op::LoadConst { dst, idx });
                Ok(dst)
            }
            Expr::Boolean(b) => {
                let dst = self.alloc_reg();
                self.emit(Op::LoadBool { dst, val: *b });
                Ok(dst)
            }
            Expr::Null => {
                let dst = self.alloc_reg();
                self.emit(Op::LoadNull { dst });
                Ok(dst)
            }
            Expr::Undefined => {
                let dst = self.alloc_reg();
                self.emit(Op::LoadUndefined { dst });
                Ok(dst)
            }
            Expr::Identifier(name) => {
                let dst = self.alloc_reg();
                let name_idx = self.add_name(name);
                self.emit(Op::GetVar { dst, name_idx });
                Ok(dst)
            }
            Expr::Binary { op, left, right } => {
                let a = self.compile_expr(left)?;
                let b = self.compile_expr(right)?;
                let dst = self.alloc_reg();
                let instr = match op {
                    BinOp::Add => Op::Add { dst, a, b },
                    BinOp::Sub => Op::Sub { dst, a, b },
                    BinOp::Mul => Op::Mul { dst, a, b },
                    BinOp::Div => Op::Div { dst, a, b },
                    BinOp::Mod => Op::Mod { dst, a, b },
                    BinOp::Exp => Op::Exp { dst, a, b },
                    BinOp::Eq | BinOp::Neq => Op::Eq { dst, a, b },
                    BinOp::StrictEq | BinOp::StrictNeq => Op::StrictEq { dst, a, b },
                    BinOp::Lt => Op::Lt { dst, a, b },
                    BinOp::LtEq => Op::LtEq { dst, a, b },
                    BinOp::Gt => Op::Gt { dst, a, b },
                    BinOp::GtEq => Op::GtEq { dst, a, b },
                    BinOp::BitAnd => Op::BitAnd { dst, a, b },
                    BinOp::BitOr => Op::BitOr { dst, a, b },
                    BinOp::BitXor => Op::BitXor { dst, a, b },
                    BinOp::Shl => Op::Shl { dst, a, b },
                    BinOp::Shr => Op::Shr { dst, a, b },
                    BinOp::UShr => Op::UShr { dst, a, b },
                    _ => Op::LoadUndefined { dst }, // TODO: And, Or, Nullish as short-circuit
                };
                self.emit(instr);
                Ok(dst)
            }
            Expr::Unary { op, operand } => {
                let src = self.compile_expr(operand)?;
                let dst = self.alloc_reg();
                match op {
                    UnaryOp::Neg => {
                        self.emit(Op::Neg { dst, src });
                    }
                    UnaryOp::Not => {
                        self.emit(Op::Not { dst, src });
                    }
                    UnaryOp::BitNot => {
                        self.emit(Op::BitNot { dst, src });
                    }
                    _ => {
                        self.emit(Op::Move { dst, src });
                    }
                }
                Ok(dst)
            }
            Expr::Assign { target, value } => {
                let v = self.compile_expr(value)?;
                match target.as_ref() {
                    Expr::Identifier(name) => {
                        let name_idx = self.add_name(name);
                        self.emit(Op::SetVar { name_idx, src: v });
                    }
                    Expr::Member { object, property } => {
                        let obj = self.compile_expr(object)?;
                        let key_idx = self.add_name(property);
                        self.emit(Op::SetProp {
                            obj,
                            key_idx,
                            value: v,
                        });
                    }
                    _ => {}
                }
                Ok(v)
            }
            Expr::Call { callee, args } => {
                // Detect member-call: obj.method(args) → CallMethod with this=obj
                if let Expr::Member { object, property } = callee.as_ref() {
                    let obj_r = self.compile_expr(object)?;
                    let method_idx = self.add_name(property);
                    let args_start = self.next_reg;
                    for arg in args {
                        let r = self.compile_expr(arg)?;
                        if r != self.next_reg - 1 {
                            let dst = self.alloc_reg();
                            self.emit(Op::Move { dst, src: r });
                        }
                    }
                    let dst = self.alloc_reg();
                    self.emit(Op::CallMethod {
                        dst,
                        obj: obj_r,
                        method_idx,
                        args_start,
                        argc: args.len() as u16,
                    });
                    return Ok(dst);
                }

                let callee_r = self.compile_expr(callee)?;
                let args_start = self.next_reg;
                for arg in args {
                    let r = self.compile_expr(arg)?;
                    // Ensure args are contiguous
                    if r != self.next_reg - 1 {
                        let dst = self.alloc_reg();
                        self.emit(Op::Move { dst, src: r });
                    }
                }
                let dst = self.alloc_reg();
                self.emit(Op::Call {
                    dst,
                    callee: callee_r,
                    args_start,
                    argc: args.len() as u16,
                });
                Ok(dst)
            }
            Expr::Member { object, property } => {
                let obj = self.compile_expr(object)?;
                let dst = self.alloc_reg();
                let key_idx = self.add_name(property);
                self.emit(Op::GetProp { dst, obj, key_idx });
                Ok(dst)
            }
            Expr::Typeof(expr) => {
                let src = self.compile_expr(expr)?;
                let dst = self.alloc_reg();
                self.emit(Op::Typeof { dst, src });
                Ok(dst)
            }
            _ => {
                // Fallback for not-yet-compiled expressions
                let dst = self.alloc_reg();
                self.emit(Op::LoadUndefined { dst });
                Ok(dst)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::parser;
    use super::*;

    fn compile(src: &str) -> Chunk {
        let stmts = parser::parse(src).unwrap();
        Compiler::compile_program(&stmts).unwrap()
    }

    #[test]
    fn compile_number_literal() {
        let chunk = compile("42");
        assert!(!chunk.ops.is_empty());
        assert!(matches!(&chunk.ops[0], Op::LoadConst { .. }));
        assert!(matches!(&chunk.constants[0], Constant::Number(42.0)));
    }

    #[test]
    fn compile_addition() {
        let chunk = compile("1 + 2");
        // LoadConst, LoadConst, Add, Return
        assert!(chunk.ops.len() >= 3);
        assert!(chunk.ops.iter().any(|op| matches!(op, Op::Add { .. })));
    }

    #[test]
    fn compile_var_decl() {
        let chunk = compile("let x = 10");
        assert!(chunk.names.contains(&"x".to_string()));
        assert!(chunk.ops.iter().any(|op| matches!(op, Op::DeclVar { .. })));
    }

    #[test]
    fn compile_if_statement() {
        let chunk = compile("if (true) { 1 } else { 2 }");
        assert!(
            chunk
                .ops
                .iter()
                .any(|op| matches!(op, Op::JumpIfFalse { .. }))
        );
    }

    #[test]
    fn constant_dedup() {
        let chunk = compile("1 + 1");
        // The constant 1.0 should appear only once
        assert_eq!(chunk.constants.len(), 1);
    }
}

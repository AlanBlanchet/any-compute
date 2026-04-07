
// ═══════════════════════════════════════════════════════════════════════════
// ── AST ─────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Expression node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    String(String),
    Boolean(bool),
    Null,
    Undefined,
    Identifier(String),
    This,

    /// Binary operation: `left op right`.
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary prefix: `op expr`.
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Postfix: `expr++` / `expr--`.
    Postfix {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Assignment: `target = value`.
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    /// Compound assignment: `target += value` etc.
    CompoundAssign {
        op: BinOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },

    /// Property access: `obj.prop`.
    Member {
        object: Box<Expr>,
        property: String,
    },
    /// Computed access: `obj[expr]`.
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    /// Function call: `callee(args...)`.
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// `new Ctor(args...)`.
    New {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Array literal: `[a, b, c]`.
    Array(Vec<Expr>),
    /// Object literal: `{ key: value, ... }`.
    Object(Vec<(String, Expr)>),
    /// Arrow function: `(params) => body`.
    Arrow {
        params: Vec<String>,
        body: Box<Stmt>,
    },
    /// Function expression: `function name?(params) { body }`.
    FunctionExpr {
        name: Option<String>,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    /// Conditional: `cond ? then : else`.
    Conditional {
        condition: Box<Expr>,
        consequent: Box<Expr>,
        alternate: Box<Expr>,
    },
    /// `typeof expr`.
    Typeof(Box<Expr>),
    /// Spread: `...expr`.
    Spread(Box<Expr>),
}

/// Statement node.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Expression statement (expression followed by `;`).
    Expr(Expr),
    /// `let`/`const`/`var` declaration.
    VarDecl {
        kind: VarKind,
        name: String,
        init: Option<Expr>,
    },
    /// Block: `{ stmts... }`.
    Block(Vec<Stmt>),
    /// `if (cond) consequent else alternate`.
    If {
        condition: Expr,
        consequent: Box<Stmt>,
        alternate: Option<Box<Stmt>>,
    },
    /// `while (cond) body`.
    While {
        condition: Expr,
        body: Box<Stmt>,
    },
    /// `for (let x of iterable) body`.
    ForOf {
        binding: String,
        iterable: Expr,
        body: Box<Stmt>,
    },
    /// `return expr?`.
    Return(Option<Expr>),
    Break,
    Continue,
    /// `function name(params) { body }`.
    FunctionDecl {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    /// `throw expr`.
    Throw(Expr),
    /// `try { body } catch(e) { handler } finally { finalizer }`.
    TryCatch {
        body: Vec<Stmt>,
        catch_binding: Option<String>,
        catch_body: Option<Vec<Stmt>>,
        finally_body: Option<Vec<Stmt>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    Let,
    Const,
    Var,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Exp,
    Eq,
    Neq,
    StrictEq,
    StrictNeq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Nullish,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    UShr,
    In,
    Instanceof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Typeof,
    Void,
    Delete,
    Inc,
    Dec,
}


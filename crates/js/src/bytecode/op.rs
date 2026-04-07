// ═══════════════════════════════════════════════════════════════════════════
// ── Instruction set ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A single bytecode instruction — register-addressed.
///
/// `Reg` = `u16` register index. Constants and strings are pooled.
pub type Reg = u16;

#[derive(Debug, Clone)]
pub enum Op {
    /// `dst = constants[idx]`
    LoadConst {
        dst: Reg,
        idx: u16,
    },
    /// `dst = undefined`
    LoadUndefined {
        dst: Reg,
    },
    /// `dst = null`
    LoadNull {
        dst: Reg,
    },
    /// `dst = bool`
    LoadBool {
        dst: Reg,
        val: bool,
    },

    /// `dst = src`
    Move {
        dst: Reg,
        src: Reg,
    },

    // ── Arithmetic ──────────────────────────────────────
    Add {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Sub {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Mul {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Div {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Mod {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Exp {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Neg {
        dst: Reg,
        src: Reg,
    },

    // ── Bitwise ─────────────────────────────────────────
    BitAnd {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    BitOr {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    BitXor {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Shl {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Shr {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    UShr {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    BitNot {
        dst: Reg,
        src: Reg,
    },

    // ── Comparison / logic ──────────────────────────────
    Eq {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    StrictEq {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Lt {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    LtEq {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Gt {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    GtEq {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Not {
        dst: Reg,
        src: Reg,
    },

    // ── Variables ───────────────────────────────────────
    /// `dst = env[name_idx]` — load from scope chain.
    GetVar {
        dst: Reg,
        name_idx: u16,
    },
    /// `env[name_idx] = src` — set in scope chain.
    SetVar {
        name_idx: u16,
        src: Reg,
    },
    /// Declare a new variable in the current scope.
    DeclVar {
        name_idx: u16,
        src: Reg,
    },

    // ── Properties ──────────────────────────────────────
    /// `dst = obj[key]` — dynamic property access.
    GetProp {
        dst: Reg,
        obj: Reg,
        key_idx: u16,
    },
    /// `obj[key] = value`
    SetProp {
        obj: Reg,
        key_idx: u16,
        value: Reg,
    },
    /// `dst = obj[index_reg]` — computed access.
    GetIndex {
        dst: Reg,
        obj: Reg,
        index: Reg,
    },
    /// `obj[index_reg] = value`
    SetIndex {
        obj: Reg,
        index: Reg,
        value: Reg,
    },

    // ── Control flow ────────────────────────────────────
    /// Unconditional jump.
    Jump {
        offset: i32,
    },
    /// Jump if register is falsy.
    JumpIfFalse {
        src: Reg,
        offset: i32,
    },
    /// Jump if register is truthy.
    JumpIfTrue {
        src: Reg,
        offset: i32,
    },

    // ── Functions / calls ───────────────────────────────
    /// `dst = call(callee, args[0..argc])`
    /// Args must be in consecutive registers starting at `args_start`.
    Call {
        dst: Reg,
        callee: Reg,
        args_start: Reg,
        argc: u16,
    },
    /// `dst = obj.method(args[0..argc])` — resolves property and calls with
    /// `this = obj`. Emitted for `a.b(c)` member-call expressions.
    CallMethod {
        dst: Reg,
        obj: Reg,
        method_idx: u16,
        args_start: Reg,
        argc: u16,
    },
    /// Return from the current function.
    Return {
        src: Reg,
    },

    // ── Object / array construction ─────────────────────
    /// `dst = new Object()` with N properties to follow.
    NewObject {
        dst: Reg,
    },
    /// `dst = new Array(elems[0..count])`
    NewArray {
        dst: Reg,
        start: Reg,
        count: u16,
    },

    // ── Special ─────────────────────────────────────────
    /// `dst = typeof src`
    Typeof {
        dst: Reg,
        src: Reg,
    },
}


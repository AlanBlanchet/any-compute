// ═══════════════════════════════════════════════════════════════════════════
// ── Token ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Token kind with optional payload for literals.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ── Literals ────────────────────────────────────────
    Number(f64),
    String(String),
    Identifier(String),
    Boolean(bool),
    Null,

    // ── Keywords ────────────────────────────────────────
    Let,
    Const,
    Var,
    Function,
    Return,
    If,
    Else,
    While,
    For,
    Break,
    Continue,
    New,
    This,
    Typeof,
    Void,
    Delete,
    In,
    Of,
    Instanceof,
    Class,
    Extends,
    Super,
    Switch,
    Case,
    Default,
    Throw,
    Try,
    Catch,
    Finally,
    Undefined,

    // ── Operators ───────────────────────────────────────
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar, // **
    Eq,       // =
    EqEq,     // ==
    EqEqEq,   // ===
    BangEq,   // !=
    BangEqEq, // !==
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,     // &&
    Or,      // ||
    Nullish, // ??
    Bang,    // !
    Tilde,   // ~
    Amp,     // &
    Pipe,    // |
    Caret,   // ^
    Shl,     // <<
    Shr,     // >>
    UShr,    // >>>
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    PlusPlus,   // ++
    MinusMinus, // --
    Arrow,      // =>
    Dot,
    DotDotDot, // ...
    Optional,  // ?.
    Question,  // ?

    // ── Delimiters ──────────────────────────────────────
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semicolon,
    Comma,
    Colon,

    // ── Special ─────────────────────────────────────────
    Eof,
}


//! JavaScript lexer — source text → token stream.
//!
//! Single-pass scanner with no allocations for keyword/operator tokens.
//! String and number literals allocate only for their content.
//!
//! ## Design
//!
//! The lexer is intentionally simple: it handles the subset of JS that
//! our engine supports (ES2020-ish without generators, async, or decorators).
//! Each `next()` call advances and returns one `Token`.

use super::JsError;


mod token;
pub use token::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Lexer ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Single-pass JavaScript lexer.
pub struct Lexer<'src> {
    src: &'src [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        Self {
            src: source.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// Current position (for error messages).
    pub fn position(&self) -> (usize, usize) {
        (self.line, self.col)
    }

    fn peek(&self) -> u8 {
        if self.pos < self.src.len() {
            self.src[self.pos]
        } else {
            0
        }
    }

    fn peek_at(&self, offset: usize) -> u8 {
        let idx = self.pos + offset;
        if idx < self.src.len() {
            self.src[idx]
        } else {
            0
        }
    }

    fn advance(&mut self) -> u8 {
        let ch = self.peek();
        if ch == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        self.pos += 1;
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Whitespace
            while self.pos < self.src.len() && self.peek().is_ascii_whitespace() {
                self.advance();
            }
            // Single-line comment
            if self.peek() == b'/' && self.peek_at(1) == b'/' {
                while self.pos < self.src.len() && self.peek() != b'\n' {
                    self.advance();
                }
                continue;
            }
            // Multi-line comment
            if self.peek() == b'/' && self.peek_at(1) == b'*' {
                self.advance();
                self.advance();
                while self.pos < self.src.len() {
                    if self.peek() == b'*' && self.peek_at(1) == b'/' {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
                continue;
            }
            break;
        }
    }

    /// Advance and return the next token.
    pub fn next(&mut self) -> Result<Token, JsError> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.src.len() {
            return Ok(Token::Eof);
        }

        let ch = self.peek();

        // Number literal
        if ch.is_ascii_digit() || (ch == b'.' && self.peek_at(1).is_ascii_digit()) {
            return self.read_number();
        }

        // String literal
        if ch == b'\'' || ch == b'"' || ch == b'`' {
            return self.read_string();
        }

        // Identifier / keyword
        if ch.is_ascii_alphabetic() || ch == b'_' || ch == b'$' {
            return self.read_identifier();
        }

        // Operators and punctuation
        self.read_operator()
    }

    fn read_number(&mut self) -> Result<Token, JsError> {
        let start = self.pos;
        // Hex
        if self.peek() == b'0' && (self.peek_at(1) == b'x' || self.peek_at(1) == b'X') {
            self.advance();
            self.advance();
            while self.peek().is_ascii_hexdigit() {
                self.advance();
            }
            let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
            let val = u64::from_str_radix(&s[2..], 16).unwrap_or(0) as f64;
            return Ok(Token::Number(val));
        }
        while self.peek().is_ascii_digit() {
            self.advance();
        }
        if self.peek() == b'.' && self.peek_at(1).is_ascii_digit() {
            self.advance();
            while self.peek().is_ascii_digit() {
                self.advance();
            }
        }
        // Exponent
        if self.peek() == b'e' || self.peek() == b'E' {
            self.advance();
            if self.peek() == b'+' || self.peek() == b'-' {
                self.advance();
            }
            while self.peek().is_ascii_digit() {
                self.advance();
            }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
        Ok(Token::Number(s.parse().unwrap_or(f64::NAN)))
    }

    fn read_string(&mut self) -> Result<Token, JsError> {
        let quote = self.advance();
        let mut buf = String::new();
        loop {
            if self.pos >= self.src.len() {
                return Err(JsError::Syntax("unterminated string".into()));
            }
            let ch = self.advance();
            if ch == quote {
                break;
            }
            if ch == b'\\' {
                let esc = self.advance();
                match esc {
                    b'n' => buf.push('\n'),
                    b't' => buf.push('\t'),
                    b'r' => buf.push('\r'),
                    b'\\' => buf.push('\\'),
                    b'\'' => buf.push('\''),
                    b'"' => buf.push('"'),
                    b'`' => buf.push('`'),
                    b'0' => buf.push('\0'),
                    _ => {
                        buf.push('\\');
                        buf.push(esc as char);
                    }
                }
            } else {
                buf.push(ch as char);
            }
        }
        Ok(Token::String(buf))
    }

    fn read_identifier(&mut self) -> Result<Token, JsError> {
        let start = self.pos;
        while self.peek().is_ascii_alphanumeric() || self.peek() == b'_' || self.peek() == b'$' {
            self.advance();
        }
        let word = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        Ok(match word {
            "let" => Token::Let,
            "const" => Token::Const,
            "var" => Token::Var,
            "function" => Token::Function,
            "return" => Token::Return,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "for" => Token::For,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "new" => Token::New,
            "this" => Token::This,
            "typeof" => Token::Typeof,
            "void" => Token::Void,
            "delete" => Token::Delete,
            "in" => Token::In,
            "of" => Token::Of,
            "instanceof" => Token::Instanceof,
            "class" => Token::Class,
            "extends" => Token::Extends,
            "super" => Token::Super,
            "switch" => Token::Switch,
            "case" => Token::Case,
            "default" => Token::Default,
            "throw" => Token::Throw,
            "try" => Token::Try,
            "catch" => Token::Catch,
            "finally" => Token::Finally,
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            "null" => Token::Null,
            "undefined" => Token::Undefined,
            _ => Token::Identifier(word.to_string()),
        })
    }

    fn read_operator(&mut self) -> Result<Token, JsError> {
        let ch = self.advance();
        Ok(match ch {
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b'{' => Token::LBrace,
            b'}' => Token::RBrace,
            b'[' => Token::LBracket,
            b']' => Token::RBracket,
            b';' => Token::Semicolon,
            b',' => Token::Comma,
            b':' => Token::Colon,
            b'~' => Token::Tilde,
            b'.' => {
                if self.peek() == b'.' && self.peek_at(1) == b'.' {
                    self.advance();
                    self.advance();
                    Token::DotDotDot
                } else {
                    Token::Dot
                }
            }
            b'+' => {
                if self.peek() == b'+' {
                    self.advance();
                    Token::PlusPlus
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::PlusEq
                } else {
                    Token::Plus
                }
            }
            b'-' => {
                if self.peek() == b'-' {
                    self.advance();
                    Token::MinusMinus
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::MinusEq
                } else {
                    Token::Minus
                }
            }
            b'*' => {
                if self.peek() == b'*' {
                    self.advance();
                    Token::StarStar
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::StarEq
                } else {
                    Token::Star
                }
            }
            b'/' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::SlashEq
                } else {
                    Token::Slash
                }
            }
            b'%' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::PercentEq
                } else {
                    Token::Percent
                }
            }
            b'=' => {
                if self.peek() == b'=' {
                    self.advance();
                    if self.peek() == b'=' {
                        self.advance();
                        Token::EqEqEq
                    } else {
                        Token::EqEq
                    }
                } else if self.peek() == b'>' {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Eq
                }
            }
            b'!' => {
                if self.peek() == b'=' {
                    self.advance();
                    if self.peek() == b'=' {
                        self.advance();
                        Token::BangEqEq
                    } else {
                        Token::BangEq
                    }
                } else {
                    Token::Bang
                }
            }
            b'<' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::LtEq
                } else if self.peek() == b'<' {
                    self.advance();
                    Token::Shl
                } else {
                    Token::Lt
                }
            }
            b'>' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::GtEq
                } else if self.peek() == b'>' {
                    self.advance();
                    if self.peek() == b'>' {
                        self.advance();
                        Token::UShr
                    } else {
                        Token::Shr
                    }
                } else {
                    Token::Gt
                }
            }
            b'&' => {
                if self.peek() == b'&' {
                    self.advance();
                    Token::And
                } else {
                    Token::Amp
                }
            }
            b'|' => {
                if self.peek() == b'|' {
                    self.advance();
                    Token::Or
                } else {
                    Token::Pipe
                }
            }
            b'^' => Token::Caret,
            b'?' => {
                if self.peek() == b'?' {
                    self.advance();
                    Token::Nullish
                } else if self.peek() == b'.' {
                    self.advance();
                    Token::Optional
                } else {
                    Token::Question
                }
            }
            _ => {
                return Err(JsError::Syntax(format!(
                    "unexpected character: {:?}",
                    ch as char
                )));
            }
        })
    }

    /// Tokenize the entire source into a Vec for the parser.
    pub fn tokenize(source: &str) -> Result<Vec<Token>, JsError> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next()?;
            if tok == Token::Eof {
                break;
            }
            tokens.push(tok);
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_arithmetic() {
        let tokens = Lexer::tokenize("1 + 2 * 3").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Number(1.0),
                Token::Plus,
                Token::Number(2.0),
                Token::Star,
                Token::Number(3.0),
            ]
        );
    }

    #[test]
    fn tokenize_let_binding() {
        let tokens = Lexer::tokenize("let x = 42;").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Identifier("x".into()),
                Token::Eq,
                Token::Number(42.0),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn tokenize_string() {
        let tokens = Lexer::tokenize(r#""hello\nworld""#).unwrap();
        assert_eq!(tokens, vec![Token::String("hello\nworld".into())]);
    }

    #[test]
    fn tokenize_arrow_fn() {
        let tokens = Lexer::tokenize("(x) => x + 1").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Identifier("x".into()),
                Token::RParen,
                Token::Arrow,
                Token::Identifier("x".into()),
                Token::Plus,
                Token::Number(1.0),
            ]
        );
    }

    #[test]
    fn tokenize_comparison() {
        let tokens = Lexer::tokenize("a === b !== c").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".into()),
                Token::EqEqEq,
                Token::Identifier("b".into()),
                Token::BangEqEq,
                Token::Identifier("c".into()),
            ]
        );
    }

    #[test]
    fn tokenize_hex_number() {
        let tokens = Lexer::tokenize("0xFF").unwrap();
        assert_eq!(tokens, vec![Token::Number(255.0)]);
    }

    #[test]
    fn comments_are_skipped() {
        let tokens = Lexer::tokenize("1 // comment\n+ 2 /* block */ + 3").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Number(1.0),
                Token::Plus,
                Token::Number(2.0),
                Token::Plus,
                Token::Number(3.0),
            ]
        );
    }
}

//! JavaScript parser — token stream → AST.
//!
//! Recursive-descent parser producing an AST of [`Expr`] and [`Stmt`] nodes.
//! Handles operator precedence via Pratt parsing (binding powers).
//!
//! ## Supported syntax
//!
//! - Variable declarations (`let`, `const`, `var`)
//! - Function declarations and arrow functions
//! - Control flow (`if/else`, `while`, `for..of`, `return`, `break`, `continue`)
//! - Expressions (arithmetic, comparison, logical, member access, calls)
//! - Object and array literals
//! - Try/catch/finally, throw
//! - `typeof`, `void`, `delete`, `instanceof`, `in`
//! - Template-ready (backtick strings parsed by lexer)

use super::JsError;
use super::lexer::Token;


mod ast;
pub use ast::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Parser ──────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), JsError> {
        let got = self.advance();
        if &got == expected {
            Ok(())
        } else {
            Err(JsError::Syntax(format!(
                "expected {expected:?}, got {got:?}"
            )))
        }
    }

    fn at(&self, tok: &Token) -> bool {
        self.peek() == tok
    }

    fn eat(&mut self, tok: &Token) -> bool {
        if self.at(tok) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Parse a full program (list of statements).
    pub fn parse_program(&mut self) -> Result<Vec<Stmt>, JsError> {
        let mut stmts = Vec::new();
        while !self.at(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, JsError> {
        match self.peek() {
            Token::Let => {
                self.advance();
                self.parse_var_decl(VarKind::Let)
            }
            Token::Const => {
                self.advance();
                self.parse_var_decl(VarKind::Const)
            }
            Token::Var => {
                self.advance();
                self.parse_var_decl(VarKind::Var)
            }
            Token::Function => {
                self.advance();
                self.parse_function_decl()
            }
            Token::LBrace => self.parse_block(),
            Token::If => {
                self.advance();
                self.parse_if()
            }
            Token::While => {
                self.advance();
                self.parse_while()
            }
            Token::For => {
                self.advance();
                self.parse_for()
            }
            Token::Return => {
                self.advance();
                let expr = if self.at(&Token::Semicolon)
                    || self.at(&Token::RBrace)
                    || self.at(&Token::Eof)
                {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.eat(&Token::Semicolon);
                Ok(Stmt::Return(expr))
            }
            Token::Break => {
                self.advance();
                self.eat(&Token::Semicolon);
                Ok(Stmt::Break)
            }
            Token::Continue => {
                self.advance();
                self.eat(&Token::Semicolon);
                Ok(Stmt::Continue)
            }
            Token::Throw => {
                self.advance();
                let expr = self.parse_expr()?;
                self.eat(&Token::Semicolon);
                Ok(Stmt::Throw(expr))
            }
            Token::Try => {
                self.advance();
                self.parse_try_catch()
            }
            _ => {
                let expr = self.parse_expr()?;
                self.eat(&Token::Semicolon);
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_var_decl(&mut self, kind: VarKind) -> Result<Stmt, JsError> {
        let Token::Identifier(name) = self.advance() else {
            return Err(JsError::Syntax(
                "expected identifier after let/const/var".into(),
            ));
        };
        let init = if self.eat(&Token::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.eat(&Token::Semicolon);
        Ok(Stmt::VarDecl { kind, name, init })
    }

    fn parse_function_decl(&mut self) -> Result<Stmt, JsError> {
        let Token::Identifier(name) = self.advance() else {
            return Err(JsError::Syntax("expected function name".into()));
        };
        let params = self.parse_param_list()?;
        let body = self.parse_block_body()?;
        Ok(Stmt::FunctionDecl { name, params, body })
    }

    fn parse_param_list(&mut self) -> Result<Vec<String>, JsError> {
        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        while !self.at(&Token::RParen) && !self.at(&Token::Eof) {
            if let Token::Identifier(name) = self.advance() {
                params.push(name);
            }
            self.eat(&Token::Comma);
        }
        self.expect(&Token::RParen)?;
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Stmt, JsError> {
        Ok(Stmt::Block(self.parse_block_body()?))
    }

    fn parse_block_body(&mut self) -> Result<Vec<Stmt>, JsError> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    fn parse_if(&mut self) -> Result<Stmt, JsError> {
        self.expect(&Token::LParen)?;
        let condition = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let consequent = Box::new(self.parse_stmt()?);
        let alternate = if self.eat(&Token::Else) {
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };
        Ok(Stmt::If {
            condition,
            consequent,
            alternate,
        })
    }

    fn parse_while(&mut self) -> Result<Stmt, JsError> {
        self.expect(&Token::LParen)?;
        let condition = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::While { condition, body })
    }

    fn parse_for(&mut self) -> Result<Stmt, JsError> {
        self.expect(&Token::LParen)?;
        // Only for..of for now
        let _kind = self.advance(); // let/const/var
        let Token::Identifier(binding) = self.advance() else {
            return Err(JsError::Syntax("expected binding in for..of".into()));
        };
        self.expect(&Token::Of)?;
        let iterable = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::ForOf {
            binding,
            iterable,
            body,
        })
    }

    fn parse_try_catch(&mut self) -> Result<Stmt, JsError> {
        let body = self.parse_block_body()?;
        let (catch_binding, catch_body) = if self.eat(&Token::Catch) {
            let binding = if self.eat(&Token::LParen) {
                let Token::Identifier(name) = self.advance() else {
                    return Err(JsError::Syntax("expected catch binding".into()));
                };
                self.expect(&Token::RParen)?;
                Some(name)
            } else {
                None
            };
            Some((binding, self.parse_block_body()?))
        } else {
            None
        }
        .map_or((None, None), |(b, body)| (b, Some(body)));
        let finally_body = if self.eat(&Token::Finally) {
            Some(self.parse_block_body()?)
        } else {
            None
        };
        Ok(Stmt::TryCatch {
            body,
            catch_binding,
            catch_body,
            finally_body,
        })
    }

    // ── Expression parsing (Pratt / precedence climbing) ────────────

    fn parse_expr(&mut self) -> Result<Expr, JsError> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<Expr, JsError> {
        let left = self.parse_ternary()?;

        if self.eat(&Token::Eq) {
            let value = self.parse_assignment()?;
            return Ok(Expr::Assign {
                target: Box::new(left),
                value: Box::new(value),
            });
        }

        // Compound assignment
        let op = match self.peek() {
            Token::PlusEq => Some(BinOp::Add),
            Token::MinusEq => Some(BinOp::Sub),
            Token::StarEq => Some(BinOp::Mul),
            Token::SlashEq => Some(BinOp::Div),
            Token::PercentEq => Some(BinOp::Mod),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            let value = self.parse_assignment()?;
            return Ok(Expr::CompoundAssign {
                op,
                target: Box::new(left),
                value: Box::new(value),
            });
        }

        Ok(left)
    }

    fn parse_ternary(&mut self) -> Result<Expr, JsError> {
        let cond = self.parse_nullish()?;
        if self.eat(&Token::Question) {
            let consequent = self.parse_assignment()?;
            self.expect(&Token::Colon)?;
            let alternate = self.parse_assignment()?;
            Ok(Expr::Conditional {
                condition: Box::new(cond),
                consequent: Box::new(consequent),
                alternate: Box::new(alternate),
            })
        } else {
            Ok(cond)
        }
    }

    fn parse_nullish(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_or()?;
        while self.eat(&Token::Nullish) {
            let right = self.parse_or()?;
            left = Expr::Binary {
                op: BinOp::Nullish,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_and()?;
        while self.eat(&Token::Or) {
            let right = self.parse_and()?;
            left = Expr::Binary {
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_bit_or()?;
        while self.eat(&Token::And) {
            let right = self.parse_bit_or()?;
            left = Expr::Binary {
                op: BinOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bit_or(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_bit_xor()?;
        while self.eat(&Token::Pipe) {
            let right = self.parse_bit_xor()?;
            left = Expr::Binary {
                op: BinOp::BitOr,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bit_xor(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_bit_and()?;
        while self.eat(&Token::Caret) {
            let right = self.parse_bit_and()?;
            left = Expr::Binary {
                op: BinOp::BitXor,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bit_and(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_equality()?;
        while self.eat(&Token::Amp) {
            let right = self.parse_equality()?;
            left = Expr::Binary {
                op: BinOp::BitAnd,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match self.peek() {
                Token::EqEq => BinOp::Eq,
                Token::BangEq => BinOp::Neq,
                Token::EqEqEq => BinOp::StrictEq,
                Token::BangEqEq => BinOp::StrictNeq,
                _ => break,
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                Token::Lt => BinOp::Lt,
                Token::LtEq => BinOp::LtEq,
                Token::Gt => BinOp::Gt,
                Token::GtEq => BinOp::GtEq,
                Token::Instanceof => BinOp::Instanceof,
                Token::In => BinOp::In,
                _ => break,
            };
            self.advance();
            let right = self.parse_shift()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Token::Shl => BinOp::Shl,
                Token::Shr => BinOp::Shr,
                Token::UShr => BinOp::UShr,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, JsError> {
        let mut left = self.parse_exp()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_exp()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_exp(&mut self) -> Result<Expr, JsError> {
        let base = self.parse_unary()?;
        if self.eat(&Token::StarStar) {
            let exp = self.parse_exp()?; // right-associative
            Ok(Expr::Binary {
                op: BinOp::Exp,
                left: Box::new(base),
                right: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, JsError> {
        match self.peek() {
            Token::Minus => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(e),
                })
            }
            Token::Bang => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(e),
                })
            }
            Token::Tilde => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::BitNot,
                    operand: Box::new(e),
                })
            }
            Token::Typeof => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Typeof(Box::new(e)))
            }
            Token::Void => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Void,
                    operand: Box::new(e),
                })
            }
            Token::Delete => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Delete,
                    operand: Box::new(e),
                })
            }
            Token::PlusPlus => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Inc,
                    operand: Box::new(e),
                })
            }
            Token::MinusMinus => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Dec,
                    operand: Box::new(e),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, JsError> {
        let mut expr = self.parse_call_member()?;
        loop {
            match self.peek() {
                Token::PlusPlus => {
                    self.advance();
                    expr = Expr::Postfix {
                        op: UnaryOp::Inc,
                        operand: Box::new(expr),
                    };
                }
                Token::MinusMinus => {
                    self.advance();
                    expr = Expr::Postfix {
                        op: UnaryOp::Dec,
                        operand: Box::new(expr),
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_member(&mut self) -> Result<Expr, JsError> {
        let mut expr = if self.at(&Token::New) {
            self.advance();
            let callee = self.parse_call_member()?;
            let args = if self.at(&Token::LParen) {
                self.parse_arg_list()?
            } else {
                vec![]
            };
            Expr::New {
                callee: Box::new(callee),
                args,
            }
        } else {
            self.parse_primary()?
        };

        loop {
            match self.peek() {
                Token::Dot => {
                    self.advance();
                    if let Token::Identifier(prop) = self.advance() {
                        expr = Expr::Member {
                            object: Box::new(expr),
                            property: prop,
                        };
                    } else {
                        return Err(JsError::Syntax("expected property name after '.'".into()));
                    }
                }
                Token::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(&Token::RBracket)?;
                    expr = Expr::Index {
                        object: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Token::LParen => {
                    let args = self.parse_arg_list()?;
                    expr = Expr::Call {
                        callee: Box::new(expr),
                        args,
                    };
                }
                Token::Optional => {
                    self.advance();
                    if let Token::Identifier(prop) = self.advance() {
                        expr = Expr::Member {
                            object: Box::new(expr),
                            property: prop,
                        };
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_arg_list(&mut self) -> Result<Vec<Expr>, JsError> {
        self.expect(&Token::LParen)?;
        let mut args = Vec::new();
        while !self.at(&Token::RParen) && !self.at(&Token::Eof) {
            if self.at(&Token::DotDotDot) {
                self.advance();
                args.push(Expr::Spread(Box::new(self.parse_assignment()?)));
            } else {
                args.push(self.parse_assignment()?);
            }
            self.eat(&Token::Comma);
        }
        self.expect(&Token::RParen)?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, JsError> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(Expr::Number(n))
            }
            Token::String(s) => {
                self.advance();
                Ok(Expr::String(s))
            }
            Token::Boolean(b) => {
                self.advance();
                Ok(Expr::Boolean(b))
            }
            Token::Null => {
                self.advance();
                Ok(Expr::Null)
            }
            Token::Undefined => {
                self.advance();
                Ok(Expr::Undefined)
            }
            Token::This => {
                self.advance();
                Ok(Expr::This)
            }
            Token::Identifier(_) => self.parse_identifier_or_arrow(),
            Token::LParen => self.parse_paren_or_arrow(),
            Token::LBracket => self.parse_array_literal(),
            Token::LBrace => self.parse_object_literal(),
            Token::Function => {
                self.advance();
                self.parse_function_expr()
            }
            tok => Err(JsError::Syntax(format!("unexpected token: {tok:?}"))),
        }
    }

    fn parse_identifier_or_arrow(&mut self) -> Result<Expr, JsError> {
        let Token::Identifier(name) = self.advance() else {
            unreachable!()
        };
        // Single-param arrow: `x => body`
        if self.at(&Token::Arrow) {
            self.advance();
            let body = if self.at(&Token::LBrace) {
                let stmts = self.parse_block_body()?;
                Stmt::Block(stmts)
            } else {
                Stmt::Return(Some(self.parse_assignment()?))
            };
            Ok(Expr::Arrow {
                params: vec![name],
                body: Box::new(body),
            })
        } else {
            Ok(Expr::Identifier(name))
        }
    }

    fn parse_paren_or_arrow(&mut self) -> Result<Expr, JsError> {
        // Look ahead to distinguish `(expr)` from `(a, b) => body`
        let save = self.pos;
        self.advance(); // consume '('

        // Try to parse as param list for arrow
        let mut params = Vec::new();
        let mut could_be_arrow = true;
        while !self.at(&Token::RParen) && !self.at(&Token::Eof) {
            if let Token::Identifier(name) = self.peek().clone() {
                self.advance();
                params.push(name);
                if !self.eat(&Token::Comma) && !self.at(&Token::RParen) {
                    could_be_arrow = false;
                    break;
                }
            } else {
                could_be_arrow = false;
                break;
            }
        }

        if could_be_arrow && self.eat(&Token::RParen) && self.at(&Token::Arrow) {
            self.advance(); // consume '=>'
            let body = if self.at(&Token::LBrace) {
                Stmt::Block(self.parse_block_body()?)
            } else {
                Stmt::Return(Some(self.parse_assignment()?))
            };
            return Ok(Expr::Arrow {
                params,
                body: Box::new(body),
            });
        }

        // Not an arrow — backtrack and parse as grouped expression
        self.pos = save;
        self.advance(); // consume '('
        let expr = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        Ok(expr)
    }

    fn parse_array_literal(&mut self) -> Result<Expr, JsError> {
        self.advance(); // '['
        let mut elems = Vec::new();
        while !self.at(&Token::RBracket) && !self.at(&Token::Eof) {
            if self.at(&Token::DotDotDot) {
                self.advance();
                elems.push(Expr::Spread(Box::new(self.parse_assignment()?)));
            } else {
                elems.push(self.parse_assignment()?);
            }
            self.eat(&Token::Comma);
        }
        self.expect(&Token::RBracket)?;
        Ok(Expr::Array(elems))
    }

    fn parse_object_literal(&mut self) -> Result<Expr, JsError> {
        self.advance(); // '{'
        let mut props = Vec::new();
        while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
            let key = match self.advance() {
                Token::Identifier(s) | Token::String(s) => s,
                Token::Number(n) => n.to_string(),
                tok => {
                    return Err(JsError::Syntax(format!(
                        "expected property key, got {tok:?}"
                    )));
                }
            };
            self.expect(&Token::Colon)?;
            let value = self.parse_assignment()?;
            props.push((key, value));
            self.eat(&Token::Comma);
        }
        self.expect(&Token::RBrace)?;
        Ok(Expr::Object(props))
    }

    fn parse_function_expr(&mut self) -> Result<Expr, JsError> {
        let name = if let Token::Identifier(_) = self.peek() {
            if let Token::Identifier(n) = self.advance() {
                Some(n)
            } else {
                None
            }
        } else {
            None
        };
        let params = self.parse_param_list()?;
        let body = self.parse_block_body()?;
        Ok(Expr::FunctionExpr { name, params, body })
    }
}

/// Parse a source string into an AST.
pub fn parse(source: &str) -> Result<Vec<Stmt>, JsError> {
    let tokens = super::lexer::Lexer::tokenize(source)?;
    Parser::new(tokens).parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_var_decl() {
        let stmts = parse("let x = 42;").unwrap();
        assert_eq!(stmts.len(), 1);
        assert!(
            matches!(&stmts[0], Stmt::VarDecl { kind: VarKind::Let, name, init: Some(Expr::Number(42.0)) } if name == "x")
        );
    }

    #[test]
    fn parse_binary_expr() {
        let stmts = parse("1 + 2 * 3").unwrap();
        // Should be Add(1, Mul(2, 3)) due to precedence
        assert!(matches!(
            &stmts[0],
            Stmt::Expr(Expr::Binary { op: BinOp::Add, .. })
        ));
    }

    #[test]
    fn parse_if_else() {
        let stmts = parse("if (x) { 1 } else { 2 }").unwrap();
        assert!(matches!(
            &stmts[0],
            Stmt::If {
                alternate: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn parse_function_decl() {
        let stmts = parse("function add(a, b) { return a + b; }").unwrap();
        assert!(
            matches!(&stmts[0], Stmt::FunctionDecl { name, params, .. } if name == "add" && params.len() == 2)
        );
    }

    #[test]
    fn parse_arrow_function() {
        let stmts = parse("let inc = (x) => x + 1").unwrap();
        assert!(matches!(
            &stmts[0],
            Stmt::VarDecl {
                init: Some(Expr::Arrow { .. }),
                ..
            }
        ));
    }

    #[test]
    fn parse_array_literal() {
        let stmts = parse("[1, 2, 3]").unwrap();
        assert!(matches!(&stmts[0], Stmt::Expr(Expr::Array(elems)) if elems.len() == 3));
    }

    #[test]
    fn parse_object_literal() {
        let stmts = parse("let o = { x: 1, y: 2 }").unwrap();
        assert!(
            matches!(&stmts[0], Stmt::VarDecl { init: Some(Expr::Object(props)), .. } if props.len() == 2)
        );
    }

    #[test]
    fn parse_member_access() {
        let stmts = parse("obj.prop").unwrap();
        assert!(
            matches!(&stmts[0], Stmt::Expr(Expr::Member { property, .. }) if property == "prop")
        );
    }
}

---
name: js
description: JS engine pipeline (lexer, parser, bytecode compiler, register VM, runtime builtins)
applyTo: "crates/js/**"
---

# JS Engine Skill — `crates/js/`

V8-like JavaScript engine: Source → Lexer → Parser → Compiler → Bytecode → VM → Result.

## Modules

- `lib.rs` — `eval(source)`, `create_vm()`, `JsError`
- `value.rs` — `JsValue` universal tagged enum, `JsObject` with prototype chain
- `lexer/` — single-pass scanner (keywords, operators, literals, comments), Token enum
- `parser/` — recursive-descent with Pratt precedence → `Expr`/`Stmt` AST types + parser impl
- `bytecode/` — register-addressed `Op` enum, `Chunk` compiler with constant dedup
- `vm.rs` — `Vm` register file, scope chain, `set_host<T>`/`host<T>()` for DOM bridge
- `runtime.rs` — `install_builtins()`: console, Math, JSON. `math_unary!` macro for Math methods

## Design Decisions

- Register VM (not stack) — fewer ops, maps to machine registers for future JIT
- `CallMethod` opcode for `a.b(c)` patterns — element methods receive correct `this`
- Compatible, not conformant — implements JS features useful for DOM/compute APIs
- `define_native_fn(name, fn)` / `define_global(name, value)` for Rust→JS bridging

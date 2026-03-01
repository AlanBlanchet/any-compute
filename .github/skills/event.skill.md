---
name: event
description: Input event model, propagation phases, and framework-agnostic handling
applyTo: "crates/core/src/interaction.rs"
---

# Event Model

## Propagation

- Three-phase propagation: **Capture → Target → Bubble** — mirrors the W3C DOM model exactly.
- `EventContext` wraps every event; call `.stop_propagation()` / `.prevent_default()` to control flow.
- Never implement custom propagation outside `EventContext` — extend it instead.

## Input coverage

- `InputEvent` is the single enum covering pointer, keyboard, focus/blur, and scroll events.
- `Modifiers` tracks shift/ctrl/alt/meta — included in every input event.

## No framework deps

- Core event model has zero UI framework dependencies — dioxus/DOM adapters live in `crates/rsx/`.
- To add a new input source: add a variant to `InputEvent`, not a new type.

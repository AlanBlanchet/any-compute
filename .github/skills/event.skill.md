---
name: event
description: Input event model, propagation phases, hover/focus tracking, and framework-agnostic dispatch
applyTo: "crates/core/**,crates/dom/**"
---

# Event Model

## Propagation

- Three-phase: Capture → Target → Bubble (W3C model)
- `EventContext` wraps every event: `.stop_propagation()` / `.prevent_default()`
- `Tree::dispatch(event) → DispatchResult` — full propagation, returns tag chain

## Input Coverage

- `InputEvent` — single enum for pointer, keyboard, text input, focus/blur, scroll
- `Modifiers` tracks shift/ctrl/alt/meta
- `TextInput { text }` separate from KeyDown for IME support

## Dispatch

- `DispatchResult` carries `tags`, `stopped`, `default_prevented`, `restyled`, `cursor`
- Editable nodes: `TextInput` inserts at caret, `KeyDown` forwarded to `handle_edit_key`
- `PointerDown` on editable node auto-focuses and places caret

## State Tracking

- `HoverState` — tag-based hover across frames (tree rebuilt each frame in immediate-mode)
- `FocusState` — focused tag for keyboard dispatch
- `Tree::hit_test(pos)` — deepest node under cursor (reverse z-order)

## No framework deps

Core event model has zero UI framework dependencies.
To add a new input source: add a variant to `InputEvent`, not a new type.

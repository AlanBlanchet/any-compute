---
name: event
description: Input event model, propagation phases, hover/focus tracking, and framework-agnostic dispatch
applyTo: "crates/core/src/interaction.rs,crates/dom/src/tree.rs"
---

# Event Model

## Propagation

- Three-phase propagation: **Capture → Target → Bubble** — mirrors the W3C DOM model exactly.
- `EventContext` wraps every event; call `.stop_propagation()` / `.prevent_default()` to control flow.
- Never implement custom propagation outside `EventContext` — extend it instead.
- `Tree::dispatch(event) → DispatchResult` performs full propagation and returns the tag chain from root → target.

## DispatchResult

- `DispatchResult` carries `tags: Vec<String>` (root → target order), `stopped`, `default_prevented`.
- `.target_tag()` — deepest (innermost) tag.
- `.bubble_tags()` — iterator from innermost → outermost (bubble order).
- The host uses `DispatchResult` to decide actions; per-node handlers are a future extension point.

## Input coverage

- `InputEvent` is the single enum covering pointer, keyboard, focus/blur, and scroll events.
- `InputEvent::pos()` extracts position from pointer events; `None` for keyboard/focus.
- `Modifiers` tracks shift/ctrl/alt/meta — included in every keyboard event.

## Hover tracking

- `HoverState` tracks the currently hovered tag across frames.
- `.update(new_tag)` returns `Option<HoverDelta>` with `left` / `entered` tags.
- The host starts transitions when `HoverDelta` is emitted (fade in/out hover effects).
- Hover is tag-based (not node-based) because the tree is rebuilt every frame (immediate-mode).

## Focus tracking

- `FocusState` tracks the focused tag for keyboard dispatch.
- `.focus(tag)` moves focus and returns the previously focused tag.
- Focus is set on pointer click and cleared on Escape.

## Hit testing

- `Tree::hit_test(pos) → Option<NodeId>` — deepest node under cursor (reverse z-order).
- `Tree::click(pos)` — walk parents from hit node to find deepest tagged node (legacy helper).
- `Tree::tag_at(pos)` — same as `click` but returns `String` (owned).

## No framework deps

- Core event model has zero UI framework dependencies — dioxus/DOM adapters live in `crates/rsx/`.
- To add a new input source: add a variant to `InputEvent`, not a new type.

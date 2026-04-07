# DOM Skill — `crates/dom/`

Arena-based scene graph with flexbox layout, HTML/CSS parsers, and GPU renderer.
GPU opt-in via `features = ["gpu"]`. Keeps `core` focused on compute.

## Modules

- `tree/` — `Tree` (arena `Vec<Slot>`), layout/paint/event dispatch, `build_address_bar()`, editable text, animation types, node kinds
- `style/` — `Style` builder, `StyleOp` pre-compiled mutations, `FlexStyle` impl, re-exports `core::flex` layout types
- `parse/` — `parse(&str) → Tree` — infallible HTML scanner, value parsers (`parse_px`, `parse_dimension`, `parse_time`, `parse_angle`, `parse_shadow`, `parse_transform`, `parse_filter`, `parse_calc_expr`), inline CSS `style="..."` support via `apply_inline_css()`, HTML entity decoding (`decode_entities`), auto-inline direction heuristic
- `css/` — `StyleSheet::parse(css)` — infallible CSS parser w/ transitions, @keyframes, animations, variables, `calc()`, advanced selectors, rule types
- `page.rs` — `Page` runtime: loads HTML+CSS+JS via `Tree` + `Vm` + `StyleSheet`, browser-like DOM API
- `gpu.rs` — (gpu feature) wgpu renderer: windowed + headless, SDF rounded-rect shader, instanced draw
- `harness.rs` — `TestHarness`: headless test driver (GPU + CPU-only modes)
- `scenario.rs` — `Action`/`Scenario` replay with pixel assertions (`AssertPixel`, `AssertRegion`)
- `theme.rs` — Catppuccin Mocha palette constants (always available, no feature gate)

## Key Patterns

### Code Compression Macros

- **`css_enums!`** — DOM-only CSS keyword enums: generates enum + Default + `from_css`/`to_css`
- **`html_tags!`** — `HtmlTag` enum + `TagKind` from compact DSL. Adding a tag = one line. Supports flags: `interactive`, `hidden`, `heading`, `inline`. `is_inline()` method checks inline flag.
- **`style_lerp_field!`** — `Style::lerp()` body from tagged field list (`[num]`/`[snap]`/`[lerp]`/`[opt_*]`)
- **`for_each_inheritable!`** — SSOT for CSS inheritable properties (feeds `gen_inherit_consts` + `do_inherit`)
- **`css_property_copy!`** — `copy_css_property()` body from `"css-name" => { fields }` table

### Named Constants

All magic numbers in `style.rs` as `pub const`: `REM_PX`, `DEFAULT_FONT_SIZE`, `DEFAULT_LINE_HEIGHT`, `CHAR_WIDTH_RATIO`, etc. GPU constants in `gpu.rs`.

### Infallible Parsers

Both `parse()` and `StyleSheet::parse()` never error or panic. Malformed input silently skipped (bad CSS declarations, unclosed tags, unknown attrs). `parse_selector` handles attribute selectors (`[attr]`), sibling combinators (`+`, `~`), commas, and unknown chars via a `chars.next()` fallback — prevents infinite loops on real-world CSS (e.g. google.com). `extract_resources` has an iteration safety guard (max = input byte count). `decode_entities` handles 50+ HTML named entities (`&amp;`, `&nbsp;`, `&eacute;`...) plus numeric (`&#NNN;`) and hex (`&#xHHH;`) forms.

### CSS → Style Pipeline

CSS text → `strip_comments` → tokenizer → `compile_declarations` (variable resolution up to 16 levels) → `RulePayload { ops, transitions, animations }` → selectors classified (simple → HashMap O(1), complex → Vec)

### Inline CSS (`style="..."`)

`apply_style_attrs()` intercepts `key == "style"` and routes to `apply_inline_css()` (in `parse.rs`). Internally delegates to the shared `compile_declaration()` pipeline (see below).

### Shared declaration pipeline

`compile_declaration(prop, value) → Vec<StyleOp>` (in `parse.rs`) is the single pipeline for compiling one CSS `property: value` pair into ops. Handles multi-op shorthands (transform, filter, box/text-shadow), longhand expansion via `expand_css_property()`, and `compile_attr()`. Used by both `apply_inline_css()` (inline styles) and `compile_declarations()` (CSS rules).

### `display:flex` → `flex-direction:row` injection

`inject_flex_row(ops)` (in `parse.rs`) detects `Display(Flex)` without a `Direction` op and pushes `Direction(Row)`. Called from both `apply_inline_css()` and `compile_declarations()` — single implementation.

### Style Builder API

Available builders: `w()`, `h()`, `wh()`, `min_w()`, `min_h()`, `row()`, `gap()`, `pad()`, `pad_xy()`, `bg()`, `border()`, `radius()`, `color()`, `cursor()`, `align()`, `align_self()`, `justify()`, `grow()`, `overflow()`, `opacity()`, `wrap()`.

**Not available**: `max_w()`, `max_h()`, `shrink()`, `display()`, `font_size()`, `position()`, `z_index()`. Set these fields directly on the struct.

**`Display` enum** (from `core::flex`): Only `Flex` (default), `Block`, `None`. No `Inline` variant — the layout engine is flex-only. Inline elements (like `<a>`, `<span>`) flow horizontally via `auto_inline_direction` post-pass: if a Box node's visible children are all inline-tagged OR text-leaf wrappers (Box with single Text child), the parent switches to `direction: Row; flex-wrap: Wrap; gap: 4px`.

**`corner_radius`**: Single `f64` — no per-corner border radius. Use `.radius(r)` builder.

**`Color` import**: NOT re-exported from `any_compute_dom::style`. Import from `any_compute_core::render::Color`.

### Cascade & Specificity

CSS specificity `(a,b,c,d)`: inline > id > class > element tag. `ua.css` provides element-level defaults (lowest specificity). `sheet.resolve(tag, classes, id, inline)` applies in this order.

### Element Abstraction — `ToDom`

`"button".to_dom()` → `DomElement` with UA defaults (cursor, padding, border, etc.). `tree.add_element(parent, "button")` and `tree.add_element_with(parent, "button", |s| s.bg(RED))` use this. UA sheet lazily initialized from `ua.css`. ALWAYS prefer `add_element` for semantic HTML elements over `add_box` with manual styles.

### StyleOp Pre-Compilation

60+ variants mapping 1:1 to Style field writes. Zero string matching at apply time — compiled once from `compile_attr()`.

### Inheritance

`StyleWritten(u64)` bitmask tracks set properties. `Style::inherit_from(parent)` copies unset inheritables. Applied automatically in `Tree::add_node()`.

### Layout

Flexbox solver in `Tree::layout_node`: dual sizing context (avail vs resolve), cross-axis stretch, flex-grow/shrink, flex-wrap, justify-content variants, aspect-ratio, `Dimension::Calc` resolution. Parent determines child size.

### Paint Pipeline

Per-node order: transform → clip → box-shadow → bg+border → content → selection → outline. All colors through `apply_opacity()`.

### Hit Test & Dispatch

`Tree::dispatch(event) → DispatchResult`: Capture → Target → Bubble propagation. Tracks hover/active/focus state, triggers restyle. `Tree::cursor_at(pos)` resolves cursor by target→root walk (first non-Default wins). `Style::cursor(Cursor::Pointer)` builder marks the `written` bitmask so inheritance doesn't override.

### Animations & Transitions

`Tree::tick(dt) → TickResult` advances transitions/animations. Per-property CSS transitions with independent timing. @keyframes with iteration/direction/fill. `Style::lerp()` blends all fields.

### Dirty Tracking

`Dirty` bitflags (LAYOUT/PAINT/Z_ORDER) + `mark_dirty()` upward propagation. Layout-affecting classification via `differs_in_layout()` and `StyleOp::affects_layout()`.

### Text Selection

Double-click word select, click-drag range select via `Tree.drag_anchor`. `char_index_at()` for proportional positioning.

### Editable Text

`set_editable(id, initial)` → nodes accept `TextInput`/`KeyDown` events. Caret tracking, value/text sync.

### Page Runtime

`Page::load(html)` → extracts `<style>`/`<script>`, builds `Tree` + `StyleSheet` + `Vm`. DOM API bridge via `JsObject` element handles with `__id`. `Vm::set_host<PageHost>()` for linear ownership.

## Test Patterns

- Tests in sibling `*_tests.rs` files (`#[cfg(test)] #[path = "..."] mod tests`)
- Conformance suite: `fixtures/conformance.css/html` — 24 CSS property groups
- Visual regression: GPU render → pixel assertions (`tests/visual.rs`, `make test-visual`)
- `TestHarness::from_css_html_cpu()` for headless CPU-only testing

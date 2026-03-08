# DOM Skill — `crates/dom/`

Arena-based scene graph with flexbox layout, HTML-like parser, and CSS parser.
Lives in its own crate to keep `core` focused on compute primitives.

## Crate Structure

| Module         | Purpose                                                                                                                                                                                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tree.rs`      | `Tree` (arena `Vec<Slot>`), `Slot`, `NodeId`, layout/paint/event dispatch                                                                                                                                                                                                                             |
| `style.rs`     | `Style` (builder pattern), `StyleOp` (pre-compiled mutations), 21 CSS enums, `Dimension` (Auto/Px/Percent/Calc), `Shadow`, `Edges`, `StyleWritten` bitmask for inheritance tracking. Every style enum has `from_css(val) -> Self` for polymorphic resolution.                                         |
| `parse.rs`     | `parse(&str) → Tree` — zero-dep fault-tolerant HTML-like scanner + `compile_attr` / `parse_px` / `parse_dimension` / `parse_time` / `parse_angle` / `parse_shadow` / `parse_transform` / `parse_filter` / `parse_calc_expr` (single source of truth for attr→Style mapping and unit/value conversion) |
| `css.rs`       | `StyleSheet::parse(css) → StyleSheet` — full CSS parser with transitions, @keyframes, animations, advanced selectors, CSS variables, `calc()`. O(1) HashMap lookups for simple selectors, tree-walking for complex selectors.                                                                         |
| `ua.css`       | User-agent defaults (`include_str!` from `css.rs`) — block-level spacing, body margin, etc.                                                                                                                                                                                                           |
| `tailwind.css` | Real compiled Tailwind v3 CSS output — parsed via `StyleSheet::parse()` at test time for visual correctness verification                                                                                                                                                                              |

## Key Patterns

### Code Compression Macros

**`css_enums!`** (style.rs) — Generates 20+ CSS keyword enums from a compact DSL.
Each entry declares variants, CSS string mappings, default, and optional fallback.
Generates: enum definition, `Default`, `from_css(&str) -> Self`, `to_css(self) -> &'static str`, and alias support.
`to_css` returns the first CSS alias — used for cursor propagation, serialization, etc.
Adding a new CSS enum = one block, no boilerplate.

**`inh!`** (style.rs, inside `inherit_from()`) — One-line macro for each inheritable property:
checks parent's `written` bit and copies the value if child hasn't set it.

### Named Constants (style.rs)

All magic numbers live in one place as `pub const`:
`REM_PX` (16.0), `DEFAULT_FONT_SIZE` (14.0), `DEFAULT_LINE_HEIGHT` (1.3),
`CHAR_WIDTH_RATIO` (0.55), `TEXT_BASELINE_RATIO` (0.85), `MIN_BAR_HEIGHT` (8.0),
`BAR_TRACK_BG` (rgba 255,255,255,20), `DEFAULT_EASING` (EaseInOut).
Used everywhere in tree/parse/css — never hardcoded.

### Shared Helpers

- **`scan_css_functions(val, cb)`** (parse.rs) — Generic CSS function scanner yielding `(name, args)` pairs. Used by `parse_transform` and `parse_filter` — both are thin wrappers with match-arm-only logic.
- **`parse_filter_amount(arg)`** (parse.rs) — Parses percentage-or-number from CSS filter function args.
- **`bar_from_attrs(attrs)`** (parse.rs) — Extracts bar (fraction, fill color) from HTML attributes. Used by both `spawn_child` and `set_kind`.
- **`expand_box_decl(prefix, value, include_style)`** (css.rs) — Unified shorthand expansion for `border` and `outline`.
- **`z_sorted_children(id)`** (tree.rs) — Returns children sorted by z-index (ascending). Used by both `paint_children` and `hit_test_node`.
- **`find_tag_node(pos)`** (tree.rs) — Hit-test + walk up to find nearest tagged ancestor. Used by both `click()` and `tag_at()`.
- **`walk_ancestors(start, closure)`** (tree.rs) — Walk from node to root, calling a closure on each `Slot`. Used for setting/clearing hover, active, and any future per-path flag.
- **`ActiveTransition::from_spec(spec, from, to)`** (tree.rs) — Construct transition from `TransitionSpec` + before/after Style snapshots.

### Style Helper Methods

- **`has_transform()`** — checks 7 transform fields for non-identity values
- **`has_visible_border()`** — `border_color.a > 0 && effective_border().any_nonzero()` — encapsulates paint/hit-test border visibility check
- **`transform_rect(Rect) -> Rect`** — apply translate + center-relative scale. Used in both paint and hit-test.
- **`apply_opacity(Color) -> Color`** — pre-multiply alpha by opacity. Applied to every emitted color.
- **`transform_text(&str) -> Cow<str>`** — apply text-transform (uppercase/lowercase/capitalize/none)
- **`char_width() -> f64`** / **`text_width(&str) -> f64`** — text measurement including letter-spacing and word-spacing

### Test Extraction

Tests live in sibling `*_tests.rs` files, wired via `#[cfg(test)] #[path = "tree_tests.rs"] mod tests;`.
This keeps the main module focused on logic while preserving full `use super::*` access.
Files using this pattern: `tree.rs` → `tree_tests.rs`, `css.rs` → `css_tests.rs`.

### Fault-Tolerant Parsers

Both `parse()` and `StyleSheet::parse()` are **infallible** — they never return errors or panic.
Malformed input is silently skipped:

- CSS: unclosed braces stop parsing (keeps rules parsed so far), bad declarations skipped
- HTML: unclosed tags auto-close, unmatched close tags ignored, empty input → default root
- Markup specials skipped during tokenization via `skip_markup_special()`: `<!-- -->` comments, `<!DOCTYPE>` declarations, `<![CDATA[]]>` sections, `<?...?>` processing instructions. Helper `find_end()` does the end-marker scan.
- Bad attribute values (e.g. `w="banana"`) silently stay at default
- @keyframes with bad stop percentages silently skipped, unknown @-rules skipped

### Arena Access

All `Tree` method bodies use `self.slot(id)` / `self.slot_mut(id)` — never raw `self.arena[id.0]`.
The `arena` field is `pub` for external read access (tests, consumers), but internal methods
go through the typed accessors for consistency and future-proofing.

### CSS → Style Pipeline

CSS is the developer-facing input format; at runtime only pre-resolved `Style` structs exist.

1. `StyleSheet::parse(css)` — hand-rolled zero-dep parser, infallible
2. CSS text → `strip_comments` → inline tokenizer handles `@keyframes`, `:root` variables, regular rules
3. `compile_declarations(body, &variables)` → `RulePayload { ops, transitions, animations }`
4. Variable resolution: `var(--name, fallback)` replaced inline during compilation (up to 16 nested levels)
5. Transition/animation longhands assembled after all declarations are parsed
6. Each comma-separated selector classified: simple → HashMap, complex → `Vec<ComplexRule>`
7. `sheet.class("name")` — O(1) HashMap → `apply_ops(&mut Style, &[StyleOp])` (pure enum match)
8. Full cascade: `resolve(tag, classes, id, inline_attrs)` — tag < class < id < inline

### StyleSheet Internal Storage

```rust
pub struct StyleSheet {
    classes: HashMap<String, RulePayload>,   // .class selectors
    tags: HashMap<String, RulePayload>,      // tag selectors
    ids: HashMap<String, RulePayload>,       // #id selectors
    complex_rules: Vec<ComplexRule>,          // descendant, child, pseudo-class selectors
    keyframes: HashMap<String, Vec<Keyframe>>, // @keyframes definitions
    variables: HashMap<String, String>,      // CSS custom properties from :root / *
}
```

`RulePayload` bundles `ops: Vec<StyleOp>`, `transitions: Vec<TransitionSpec>`, `animations: Vec<AnimationSpec>`.

### CSS Transitions

Parsed via shorthand `transition: prop dur ease delay` or longhands (`transition-property`, `-duration`, `-timing-function`, `-delay`). Multi-property transitions supported via comma separation.

```rust
TransitionSpec { property: String, duration_secs: f64, easing: Easing, delay_secs: f64 }
```

Access: `sheet.class_transitions("name")` → `&[TransitionSpec]`.
Bridges to `core::animation::Transition<T>` at runtime. `Easing::from_css("ease")` maps CSS keywords.

### @keyframes + Animations

`@keyframes name { from { ... } 50% { ... } to { ... } }` parsed inline. Stops sorted by percentage.
Animation shorthand `animation: name dur ease delay count direction fill` parsed with positional heuristics (first time = duration, second = delay, etc).

```rust
Keyframe { stop: f64, ops: Vec<StyleOp> }
AnimationSpec { name, duration_secs, easing, delay_secs, iteration_count, direction, fill_mode }
AnimationIterCount::Count(f64) | ::Infinite
AnimationDirection::Normal | Reverse | Alternate | AlternateReverse
AnimationFillMode::None | Forwards | Backwards | Both
```

Access: `sheet.keyframes("name")` → `Option<&[Keyframe]>`, `sheet.class_animations("name")` → `&[AnimationSpec]`.

### CSS Custom Properties (Variables)

Extracted from `:root { --name: value; }` and `* { --name: value; }` rules.
Resolved inline during `compile_declarations` via `resolve_var()`:

- `var(--name)` → looked up in stylesheet variables
- `var(--name, fallback)` → fallback if missing
- Nested up to 16 levels
  Access: `sheet.var("--name")` → `Option<&str>`.

### calc()

`Dimension::Calc { percent: f64, px: f64 }` — covers `calc(A% ± Bpx)`.
Resolved at layout time: `parent * percent / 100 + px`.
Stays `Copy` — no heap allocation. Simplifies to `Percent` or `Px` when only one component.
Parsed via `parse_calc_expr()` in `parse.rs` — splits on whitespace-delimited `+`/`-` operators.

### Advanced Selectors

| Selector     | Example               | Storage                   |
| ------------ | --------------------- | ------------------------- |
| Class        | `.card`               | `classes` HashMap         |
| Tag          | `div`                 | `tags` HashMap            |
| Id           | `#main`               | `ids` HashMap             |
| Comma group  | `.a, .b`              | both separately           |
| Descendant   | `.parent .child`      | `complex_rules`           |
| Child        | `.parent > .child`    | `complex_rules`           |
| Pseudo-class | `.btn:hover`          | `complex_rules`           |
| Universal    | `*`                   | `tags` HashMap            |
| Compound     | `div.card#main:hover` | classified per complexity |

Selector parsing: `parse_selector(sel) → ParsedSelector` with `(Combinator, SelectorSegment)` chain.
Specificity: `(ids, classes+pseudos, tags)` — standard CSS (a,b,c) calculation.
Pseudo-classes: `Hover`, `Focus`, `Active`, `Visited`, `FirstChild`, `LastChild`, `NthChild(a, b)`.
`nth-child` supports `odd`, `even`, `an+b` syntax.

### StyleOp Pre-Compilation

`StyleOp` (60+ variants) maps 1:1 to `Style` field writes:

- Zero string matching at apply time — compiled once, applied N times
- Inheritable properties set `written` bits in `StyleWritten` bitmask on apply
- `compile_attr(key, val) → Option<StyleOp>` — single source of truth (HTML + CSS both use it)

### CSS Property Inheritance

`StyleWritten(u64)` bitmask tracks explicitly-set properties. 15 inheritable property bits:
`INHERIT_COLOR`, `INHERIT_FONT_SIZE`, `INHERIT_FONT_WEIGHT`, `INHERIT_LINE_HEIGHT`,
`INHERIT_TEXT_ALIGN`, `INHERIT_WHITE_SPACE`, `INHERIT_VISIBILITY`, `INHERIT_CURSOR`,
`INHERIT_LETTER_SPACING`, `INHERIT_WORD_SPACING`, `INHERIT_TEXT_TRANSFORM`,
`INHERIT_TEXT_INDENT`, `INHERIT_WORD_BREAK`, `INHERIT_DIRECTION`, `INHERIT_FONT_FAMILY`.

`Style::inherit_from(&mut self, parent: &Style)` copies unset inheritable properties from parent.

### Style Enums (21 total)

All implement `fn from_css(val: &str) -> Self` (or `Option<Self>` for FontWeight):

| Enum             | CSS Property      | Default       |
| ---------------- | ----------------- | ------------- |
| `Display`        | `display`         | `Flex`        |
| `Direction`      | `flex-direction`  | `Column`      |
| `FlexWrap`       | `flex-wrap`       | `NoWrap`      |
| `Align`          | `align-items`     | `Stretch`     |
| `Justify`        | `justify-content` | `Start`       |
| `Position`       | `position`        | `Relative`    |
| `Overflow`       | `overflow`        | `Visible`     |
| `TextAlign`      | `text-align`      | `Left`        |
| `Visibility`     | `visibility`      | `Visible`     |
| `WhiteSpace`     | `white-space`     | `Normal`      |
| `BoxSizing`      | `box-sizing`      | `BorderBox`   |
| `FontWeight`     | `font-weight`     | `Normal(400)` |
| `TextDecoration` | `text-decoration` | `None`        |
| `TextTransform`  | `text-transform`  | `None`        |
| `Cursor`         | `cursor`          | `Default`     |
| `PointerEvents`  | `pointer-events`  | `Auto`        |
| `UserSelect`     | `user-select`     | `Auto`        |
| `TextOverflow`   | `text-overflow`   | `Clip`        |
| `WordBreak`      | `word-break`      | `Normal`      |
| `BorderStyle`    | `border-style`    | `None`        |
| `ObjectFit`      | `object-fit`      | `Fill`        |

### Value Parsers (parse.rs)

| Parser                    | Input                                            | Output                  |
| ------------------------- | ------------------------------------------------ | ----------------------- |
| `parse_px(val)`           | `"16px"`, `"1rem"`, `"2em"`, `"14"`              | `Option<f64>` (px)      |
| `parse_dimension(val)`    | above + `"auto"`, `"50%"`, `"calc(100% - 20px)"` | `Option<Dimension>`     |
| `parse_time(val)`         | `"300ms"`, `"1.5s"`, `"0.3"`                     | `Option<f64>` (seconds) |
| `parse_angle(val)`        | `"45deg"`, `"1.5rad"`, `"0.25turn"`, `"90"`      | `Option<f64>` (degrees) |
| `parse_shadow(val)`       | `"2px 4px 6px #000"`                             | `Option<Shadow>`        |
| `parse_transform(val)`    | `"translateX(10px) rotate(45deg)"`               | `Vec<StyleOp>`          |
| `parse_filter(val)`       | `"blur(5px) brightness(120%)"`                   | `Vec<StyleOp>`          |
| `parse_line_height(val)`  | `"1.5"`, `"24px"`, `"normal"`, `"150%"`          | `Option<StyleOp>`       |
| `parse_color(val)`        | `"#rgb"`, `"#rrggbb"`, `"rgb(r,g,b)"`, named     | `Option<Color>`         |
| `parse_calc_expr(expr)`   | `"100% - 20px"`                                  | `Option<Dimension>`     |
| `parse_aspect_ratio(val)` | `"16 / 9"`, `"1.5"`                              | `Option<f64>`           |

### Unit Conversion

`parse_px(val)` is the single source of truth:

- `rem` → × `REM_PX` (16.0)
- `em` → × `REM_PX`
- `px` → strip suffix
- bare number → direct parse
- `norm_val` in css.rs delegates to `parse_px` for shorthand expansion round-trips

### Tailwind CSS

Real Tailwind v3 compiled CSS output (`tailwind.css`) parsed through `StyleSheet::parse()`.
Exported as `any_compute_dom::TAILWIND_CSS` for consumers.
Tests verify computed Style values and pixel-level visual equivalence using `PixelBuffer::diff()`.

### Layout

Flexbox-like solver in `Tree::layout_node`:

- **Dual sizing context**: `avail_w/h` = flex-allocated space (auto-width fallback), `resolve_w/h` = parent's content dimensions (percentage resolution). This prevents percentage values from double-resolving through flex allocation.
- **Cross-axis stretch**: `Align` defaults to `Stretch` (CSS spec). Children without explicit cross-dimension stretch to `child_avail_w/h`. Non-stretch alignments (`Start`/`Center`/`End`) let the child size to content (0.0 for height, `intrinsic_width` for width). Stretch is gated on `child_align_self.unwrap_or(parent.align)`.
- **Cross-axis alignment**: Center/End use `line_cross` (pre-computed actual line cross dimension), NOT `child_avail_h/w`. Auto-height containers have `child_avail_h = 0`, which would shift Center/End children above the line. The `line_cross` is the max of `(child_cross + margin_cross)` across all items in the flex line.
- **flex-basis**: Overrides width (row) or height (column) on the main axis when not `Auto`. Falls back to explicit width/height, then intrinsic measurement.
- **order**: `flow_children` are sorted by `style.order` (stable sort — preserves DOM order for ties) before layout.
- **line_height_absolute**: When `true`, `style.line_height` is in absolute px (parsed from `24px`). When `false`, it's a unitless multiplier (parsed from `1.5` or `normal`).
- **final_h respects avail_h**: When a node has no explicit height, its final height is `max(content + padding + border, avail_h)`. This propagates cross-axis stretch from the parent through `layout_node` without needing an extra parameter.
- **Main-axis intrinsic sizing**: `Tree::intrinsic_width()` recursively measures text-extent / child-sum / child-max
- **Flex-grow**: distributes remaining main-axis space after intrinsic sizing
- **Flex-shrink**: proportional shrinking only on _definite_ main axes (main_budget > 0), respecting min constraints
- **Flex-wrap**: Children broken into lines based on main-axis budget overflow. Each line independently computes grow/shrink/justify and positions children. Cross-axis cursor advances between lines. `wrap-reverse` reverses cross-axis line order.
- **justify-content**: `space-between` (gaps between items only), `space-around` (half-gap edges), `space-evenly` (equal gaps including edges) — computed per flex line.
- **aspect-ratio**: When set and height is auto, derives height from `width / aspect_ratio`.
- **white-space: nowrap**: Forces 1 line in text height calculation.
- **letter-spacing + word-spacing**: Applied via `style.text_width()` helper in both layout and intrinsic sizing.
- **Dimension::Calc resolution**: `calc(100% - 20px)` resolved during layout with actual parent sizes
- The parent determines child size, not the child's own direction

### Visual Comparison Testing

Pixel-accurate comparison against Chrome headless reference:

- **Chrome headless**: `google-chrome-stable --headless=new --screenshot=<path> --window-size=800,600 --force-device-scale-factor=1 <url>` → exact 800×600 PNG, zero decorations
- **Engine capture**: `maim -u -i <wid>` → window content; may include CSD title bar (37px on current system), crop with `convert -crop 800x600+0+37`
- **Known text offset**: Engine uses `chars × font_size × 0.55` width and `lines × font_size × line_height` height estimation. This produces ~4px vertical cumulative offset vs real font metrics. 88% exact pixel match is the current baseline.
- **Film-strip transition testing**: Freeze animations at N time steps using `animation-play-state: paused` + negative `animation-delay` in CSS, then screenshot all frames in one image. No JS/Puppeteer needed.

### Paint Pipeline

Per-node rendering order in `paint_node`: transform → clip → box-shadow → bg+border → content (text/bar) → outline.
Text features pipeline: text-transform → text-overflow ellipsis → text-align offset → text-indent → text-shadow → text-decoration (underline/overline/line-through via `Line` primitive).
All emitted colors go through `apply_opacity()`. Invisible nodes (`opacity ≤ 0` or `visibility: hidden`) skip self but still paint children.**Border on transparent bg**: The bg+border Rect is emitted when `bg.a > 0 || has_border` — borders render even when background is fully transparent (critical for outline-style buttons).

### Hit Test & Event Dispatch

- `Tree::hit_test(pos)` — depth-first, reverse z-order via `z_sorted_children()`;
  `pointer-events: none` → skip node and children; transform-aware bounds via `transform_rect()`
- `Tree::click(pos)` — `find_tag_node(pos)` walks parents to find tagged node; `tag_at(pos)` delegates to it
- `Tree::dispatch(&mut self, event)` — full Capture → Target → Bubble propagation.
  **Mutates tree**: automatically tracks hover/active state on slots and triggers restyle.
  Returns `DispatchResult { tags, stopped, default_prevented, restyled, cursor }`.
  `cursor` resolved by walking target→root, first non-default cursor wins (uses `Cursor::to_css()`).
  `restyled` is `true` only when hover/active state actually changed — callers skip repaint when `false`.
  - `PointerMove` → updates `hovered` flag via `walk_ancestors`, restyles
  - `PointerDown` → sets `active` flag via `walk_ancestors`, restyles
  - `PointerUp` → clears `active` on **all** slots (not just target path), restyles each
- `Tree::focus(id)` / `Tree::blur()` — manages `Slot.focused` flag for `:focus` pseudo-class matching
- `Tree::scroll(pos, delta)` — nearest `Overflow::Scroll` container
- Scenario replay lives in `crates/canvas/scenario.rs` — see canvas skill file

### Live Restyle (Pseudo-Classes)

When `dispatch()` changes hover/active state, it calls `restyle_path(leaf)` which walks
root→leaf re-resolving each node's style:

1. Start from `slot.base_style` (the parse-time resolved style)
2. Walk `sheet.complex_rules()` — match by element/class/pseudo-class via `matches_complex_rule()`
3. If node has CSS `TransitionSpec`s and style actually changed → create `ActiveTransition` (interpolation)
4. Otherwise snap to the new style directly

Each `Slot` stores `element: String`, `class_list: Vec<String>`, `id: Option<String>`, `hovered: bool`, `active: bool`, `focused: bool`
for runtime selector matching. `Tree` holds `sheet: Option<Arc<StyleSheet>>`, `focused: Option<NodeId>` for restyle access.

Complex selector matching walks right-to-left through `ParsedSelector.segments`:

- Final segment: match tag/class + pseudo-classes (Hover, Active, Focus, FirstChild, LastChild, NthChild)
- `NthChild(a, b)`: 1-indexed CSS an+b formula — handles a==0 (exact match) and general modular arithmetic
- ID matching: `seg.id` checked against `slot.id` (populated from HTML `id` attribute in `populate_identity`)
- Ancestor segments: walk up tree via Descendant/Child combinator
- `universal: true` on a SelectorSegment matches any element

### Animation Runtime

Per-slot animation state: `transitions: Vec<ActiveTransition>`, `animations: Vec<ActiveAnimation>`.

- `Timing` — shared struct embedded in both `ActiveTransition` and `ActiveAnimation` holding `elapsed/duration/delay/easing` with `raw_progress()`, `eased_progress()`, `started()`, `finished()`, `active_time()`
- `Tree::start_animations()` — scans all nodes, creates `ActiveAnimation` from CSS class specs + @keyframes
- `Tree::tick(dt) -> TickResult` — advances all transitions/animations, interpolates styles, auto re-layouts when `needs_layout` is set. Returns `TickResult { active, needs_layout }`
- `Tree::has_active_animations() -> bool` — query for redraw scheduling
- `ActiveTransition` — from/to Style snapshots + `timing: Timing`, one per CSS property. `progress()` delegates to `timing.eased_progress()`, `finished()` delegates to `timing.finished()`
- `ActiveAnimation` — keyframe list + `timing: Timing` + iteration count/direction/fill mode. Custom `progress()` and `finished()` build on `timing.active_time()`
- Keyframe interpolation: `apply_to()` does single-pass bracket search, clones style, applies prev/next keyframe ops independently, then `Style::lerp(local_t)`. Properties not touched by keyframes stay identical in both copies so lerp is a no-op for them.

**Per-property transitions**: `restyle_node()` creates one `ActiveTransition` per `TransitionSpec`
(each with its own duration/delay/easing). `tick()` starts from the target (`to`) style, then for
each active transition computes full lerp progress and copies only the relevant field via
`copy_css_property(dst, src, prop)`. This gives each property independent timing.
`copy_css_property` maps ~45 CSS property names to `Style` field copies — must stay in sync
with `StyleOp::apply()` and `compile_attr()` (the "three registries" pattern).

On completion, transitions are removed and `slot.style = to`. No base_style rebuild in tick —
hover/active styles are resolved exclusively by `restyle_node()` at dispatch time.

**Reverse transitions**: When hover ends mid-transition, `restyle_node()` creates a new transition
with `from = current_interpolated_style` and `to = base_style + rules`. This produces smooth
reverse animations from the exact current visual state.

### Style Interpolation (`Style::lerp`)

`Style::lerp(&self, other: &Style, t: f64) -> Style` blends every field:

- **f64 fields** (gap, opacity, border_width, corner_radius, font_size, transforms, filters, etc.) — linear interpolation
- **Color fields** (background, border_color, color, outline_color) — per-component RGBA lerp via `Lerp` trait
- **Edges** (padding, margin) — per-side lerp
- **Dimension** (width, height, min/max, offsets, flex_basis) — same-variant lerp (Px↔Px, Percent↔Percent, Calc↔Calc), mixed snap at t=0.5
- **FontWeight** — lerp the u16 value, round to nearest integer
- **Shadow** — component-wise lerp when both Some
- **Enums** (display, visibility, direction, align, etc.) — snap at t=0.5
- **StyleWritten** — union of both masks

Helper lerps also available: `Dimension::lerp()`, `Edges::lerp()`, `Shadow::lerp()`, `FontWeight::lerp()`.

### DOM Playground (`examples/dom/`)

Interactive window for testing all DOM/CSS features (run via `make dom`):

- Loads CSS/HTML from external files via `include_str!` — no embedded markup
- Animation loop: `ControlFlow::Poll` when animations active, `ControlFlow::Wait` when idle
- **dt cap**: `MAX_FRAME_DT` const (0.032) caps dt in `AboutToWait` — prevents transitions completing instantly after idle (when `Wait` mode yields multi-second dt)
- Tree handles hover/active state internally — dispatch triggers restyle
- `tree.tick(dt)` called each frame in `AboutToWait` event; auto re-layouts internally. Check `result.active` for redraw scheduling — no manual `needs_layout` flag needed
- **OS cursor**: dispatch returns `cursor` string → map to `CursorIcon` variant → `window.set_cursor_icon()`; track `current_cursor` to avoid redundant calls
- `tree.start_animations()` called after parse + layout
- **CSS variables**: All Catppuccin palette colors use `var(--blue)` etc. from `PALETTE_CSS` — zero hardcoded palette hex in playground.css
- Exercises: @keyframes, transitions, :hover/:active, flexbox, colors, borders, cursor, CSS variables
- Uses `any-compute-canvas::gpu::Gpu` for rendering

## Dependencies

- `any-compute-core` — `layout::{Rect, Point, Size}`, `render::{Color, Primitive, RenderList, Border}`,
  `animation::Easing`, `interaction::{InputEvent, EventContext, DispatchResult, Phase}`
- No external deps for the lib — all parsing is hand-rolled
- No feature flags

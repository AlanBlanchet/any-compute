# DOM Skill — `crates/dom/`

Arena-based scene graph with flexbox layout, HTML-like parser, and CSS parser.
Lives in its own crate to keep `core` focused on compute primitives.

## Crate Structure

| Module     | Purpose                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------ |
| `tree.rs`  | `Tree` (arena `Vec<Slot>`), `Slot`, `NodeId`, layout/paint/event dispatch                  |
| `style.rs` | `Style` (builder pattern), `StyleOp` (pre-compiled mutations), `Dimension`, `Direction`, `Align`, `Justify`, `Edges`, `REM_PX` const — every style enum has `from_css(val) -> Self` for polymorphic string→enum resolution |
| `parse.rs` | `parse(&str) → Tree` — zero-dep fault-tolerant HTML-like scanner + `compile_attr` / `parse_px` (single source of truth for attr→Style mapping and unit conversion) |
| `css.rs`   | `StyleSheet::parse(css) → StyleSheet` — fault-tolerant CSS subset parser, compiles to `Vec<StyleOp>`, O(1) lookups |
| `tailwind.css` | Real compiled Tailwind v3 CSS output — parsed via `StyleSheet::parse()` at test time for visual correctness verification |

## Key Patterns

### Fault-Tolerant Parsers

Both `parse()` and `StyleSheet::parse()` are **infallible** — they never return errors or panic.
Malformed input is silently skipped:

- CSS: unclosed braces stop parsing (keeps rules parsed so far), bad declarations skipped
- HTML: unclosed tags auto-close, unmatched close tags ignored, empty input → default root
- Bad attribute values (e.g. `w="banana"`) silently stay at default
- `ParseError` struct kept for external consumers but internal parsers never produce it

### Arena Access

All `Tree` method bodies use `self.slot(id)` / `self.slot_mut(id)` — never raw `self.arena[id.0]`.
The `arena` field is `pub` for external read access (tests, consumers), but internal methods
go through the typed accessors for consistency and future-proofing (e.g. bounds-checking, validity).

### Node Creation

`Tree::add_box`, `add_text`, `add_bar` — each calls the internal `add_node` helper which
allocates the Slot, sets parent/child links, and assigns Kind. `Slot::new` centralises
default construction.

### CSS → Style Pipeline

CSS is the developer-facing input format; at runtime only pre-resolved `Style` structs exist.
The pipeline compiles CSS text into `Vec<StyleOp>` at parse time — resolve-time is pure enum-match with zero string parsing.

1. `StyleSheet::parse(css)` — hand-rolled zero-dep parser, infallible
2. CSS text → `strip_comments` → `parse_rules` → `compile_declarations` → `Vec<StyleOp>` stored per class/tag/id
3. `compile_declarations` expands shorthands (`padding: V H` → 4 sides) via `expand_css_property` + `norm_val`, then delegates to `compile_attr` → `StyleOp`
4. `sheet.class("name")` — O(1) HashMap lookup → `apply_ops(&mut Style, &[StyleOp])` (pure enum-match, no string hashing)
5. `sheet.classes(&["a", "b"])` — merge multiple classes in specificity order
6. `StyleSheet::lookup()` — private helper deduplicating class/tag/id resolution
7. Full cascade: `resolve(tag, classes, id, inline_attrs)` — tag < class < id < inline

### StyleOp Pre-Compilation

`StyleOp` (37 variants) maps 1:1 to `Style` field writes. Benefits:
- Zero string matching at apply time — compiled once, applied N times
- Enum variant carries the resolved value (Color, f64, Dimension, etc.)
- `compile_attr(key, val) → Option<StyleOp>` is the single source of truth for the attr→Style mapping
- Both HTML parser and CSS engine delegate through `compile_attr` — no duplication

### Style Enum `from_css()` Polymorphism

All 11 style enums + `FontWeight` implement `fn from_css(val: &str) -> Self` (or `-> Option<Self>` for FontWeight).
This moves CSS string→enum mapping onto the type itself, eliminating 53+ fully-qualified paths in `compile_attr`.

| Enum | Default on unknown |
|------|--------------------|
| `Display` | `Flex` |
| `Direction` | `Column` |
| `FlexWrap` | `NoWrap` |
| `Align` | `Stretch` |
| `Justify` | `Start` |
| `Position` | `Relative` |
| `Overflow` | `Visible` |
| `TextAlign` | `Left` |
| `Visibility` | `Visible` |
| `WhiteSpace` | `Normal` |
| `BoxSizing` | `ContentBox` |
| `FontWeight` | `None` (returns `Option`) |

`compile_attr` calls e.g. `Display::from_css(val)` — no `crate::style::Display::Flex` inline.

### Unit Conversion

`parse_px(val) → Option<f64>` is the single source of truth for all CSS length parsing:
- `rem` → × `REM_PX` (16.0, defined in style.rs)
- `em` → × `REM_PX`
- `px` → strip suffix
- bare number → direct parse
- `parse_dimension` delegates to `parse_px` for the px path (adds `auto` and `%` on top)
- `norm_val` in css.rs delegates to `parse_px` for the string round-trip needed by shorthand expansion

### Tailwind CSS

Real Tailwind v3 compiled CSS output (`tailwind.css`) is included as `include_str!` and parsed through `StyleSheet::parse()`.
Exported as `any_compute_dom::TAILWIND_CSS` for consumers (bench window merges it with bench.css).
Tests verify both computed Style values (rem→px conversion, color hex parsing, shorthand expansion) and pixel-level visual equivalence using `PixelBuffer::diff()`.
No custom Tailwind runtime — just real CSS parsed by our engine.

### Parser Deduplication

- `compile_attr(key, val) → Option<StyleOp>` — single source of truth for attr→Style field mapping (used by HTML + CSS)
- `apply_style_attrs(&mut Style, attrs)` — iterates attrs through `compile_attr` (pub(crate), shared by parse + css)
- `parse_px(val) → Option<f64>` — single source of truth for CSS length unit conversion
- `spawn_child()` — creates a child node from tag + attrs and applies `data-tag`/`tag`
- `set_kind()` — transforms an existing node's kind (used only for the root)
- `apply_tag()` — extracts and applies `data-tag`/`tag` attribute
- `map_tag()` — canonical tag → `TagMapping` (Box / Text / Bar)

### Layout

Flexbox-like solver in `Tree::layout_node`. Key behaviours:

- **Cross-axis stretch** (CSS default `align-items: stretch`): children with no explicit width/height
  in the cross dimension get `avail_w`/`avail_h` from the parent — they stretch to fill.
- **Main-axis intrinsic sizing**: row children with no explicit width are measured via
  `Tree::intrinsic_width()` (recursive text-extent / child-sum / child-max) so they occupy
  their content width. This prevents text from collapsing to 0px in a row.
- `final_w = style.width.resolve(avail_w).unwrap_or((avail_w - margin_h).max(0.0))` — always
  stretches when the parent offers space, regardless of the node's own direction.
- Flex-grow distributes remaining main-axis space after intrinsic sizing.
- **Flex-shrink**: when children exceed available main-axis space, proportional shrinking occurs.
  Each child shrinks by `(child_flex_shrink / total_shrink) * overflow`, respecting `min_width`/`min_height` constraints.
  Default `flex_shrink` is `1.0` (CSS spec default).
- **The parent determines child size, not the child's own direction** — a row-direction node
  inside a column parent still stretches its width to fill the column.

### Hit Test & Event Dispatch

- `Tree::hit_test(pos)` — recursive depth-first, returns deepest node whose `rect.contains(pos)`.
  Children iterated in reverse for z-order (last child = on top).
- `Tree::click(pos)` — calls `hit_test`, then walks parents upward until finding a tagged node (returns `&str`).
- `Tree::tag_at(pos)` — same as `click` but returns owned `String`.
- `Tree::dispatch(event) → DispatchResult` — full Capture → Target → Bubble propagation:
  1. Hit-tests to find target (pointer events) or uses focused node (keyboard)
  2. Builds ancestor path root → target via `ancestor_path`
  3. Collects tags along path via `collect_tags`
  4. Walks capture phase (root → target-1), target phase, bubble phase (target-1 → root)
  5. Returns `DispatchResult { tags, stopped, default_prevented }`
- `Tree::scroll(pos, delta)` — walks parents to find nearest `Overflow::Scroll` container.
- Both rely on layout rects being correct — if a node has a 0-width rect, it's invisible to clicks.

### Style

Builder pattern with chaining: `Style::default().w(200.0).h(100.0).bg(Color::WHITE)`.
Every field is `pub` for direct mutation when builders are excessive.
CSS classes return fully resolved `Style` values that can be further customised via builder
or by mutating fields directly (e.g. `s("btn").color(fg)` or `btn_s.background = bg`).

## Dependencies

- `any-compute-core` — `layout::{Rect, Point, Size}`, `render::{Color, Primitive, RenderList, Border}`,
  `interaction::{InputEvent, EventContext, DispatchResult, Phase}`
- No external deps for the lib — all parsing is hand-rolled
- No feature flags — GPU rendering moved to `crates/bench/`

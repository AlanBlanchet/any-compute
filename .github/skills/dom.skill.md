# DOM Skill — `crates/dom/`

Arena-based scene graph with flexbox layout, HTML-like parser, and CSS parser.
Lives in its own crate to keep `core` focused on compute primitives.

## Crate Structure

| Module     | Purpose                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------ |
| `tree.rs`  | `Tree` (arena `Vec<Slot>`), `Slot`, `NodeId`, layout/paint/event dispatch                  |
| `style.rs` | `Style` (builder pattern), `Dimension`, `Direction`, `Align`, `Justify`, `Edges`           |
| `parse.rs` | `parse(&str) → Tree` — zero-dep fault-tolerant HTML-like scanner + tree builder            |
| `css.rs`   | `StyleSheet::parse(css) → StyleSheet` — fault-tolerant CSS subset parser with O(1) lookups |

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

1. `StyleSheet::parse(css)` — hand-rolled zero-dep parser, **returns `Self` directly** (infallible)
2. `sheet.class("name")` — O(1) HashMap lookup → `apply_style_attrs` → `Style`
3. `sheet.classes(&["a", "b"])` — merge multiple classes in specificity order
4. Property normalization: CSS names (`flex-direction`, `align-items`, `background-color`) → our attr names
5. Shorthand expansion: `padding: V H` → individual sides
6. px suffix stripped automatically; `transparent` handled as a color keyword

### CSS + HTML Integration

- `parse_with_css(html, &sheet)` → `Tree` (infallible, no `.unwrap()` needed)
- Specificity order: tag rules < class rules < id rules < inline attrs
- `apply_style_attrs(&mut Style, attrs)` is shared between HTML parser and CSS resolver

### Parser Deduplication

- `spawn_child()` — creates a child node from tag + attrs and applies `data-tag`/`tag`
- `set_kind()` — transforms an existing node's kind (used only for the root)
- `apply_tag()` — extracts and applies `data-tag`/`tag` attribute
- `apply_style_attrs()` — maps attribute key/value pairs onto `&mut Style` (pub(crate), shared by parse + css)
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
- **The parent determines child size, not the child's own direction** — a row-direction node
  inside a column parent still stretches its width to fill the column.

### Hit Test & Click Dispatch

- `Tree::hit_test(pos)` — recursive depth-first, returns deepest node whose `rect.contains(pos)`.
  Children iterated in reverse for z-order (last child = on top).
- `Tree::click(pos)` — calls `hit_test`, then walks parents upward until finding a tagged node.
- Both rely on layout rects being correct — if a node has a 0-width rect, it's invisible to clicks.

### Style

Builder pattern with chaining: `Style::default().w(200.0).h(100.0).bg(Color::WHITE)`.
Every field is `pub` for direct mutation when builders are excessive.
CSS classes return fully resolved `Style` values that can be further customised via builder
or by mutating fields directly (e.g. `s("btn").color(fg)` or `btn_s.background = bg`).

## Dependencies

- `any-compute-core` — `layout::{Rect, Point, Size}`, `render::{Color, Primitive, RenderList, Border}`,
  `interaction::{InputEvent, EventContext, EventResponse, Phase}`
- No external deps for the lib — all parsing is hand-rolled
- No feature flags — GPU rendering moved to `crates/bench/`

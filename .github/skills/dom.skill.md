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
| `tailwind.css` | Real compiled Tailwind v3 CSS output — parsed via `StyleSheet::parse()` at test time for visual correctness verification                                                                                                                                                                              |

## Key Patterns

### Fault-Tolerant Parsers

Both `parse()` and `StyleSheet::parse()` are **infallible** — they never return errors or panic.
Malformed input is silently skipped:

- CSS: unclosed braces stop parsing (keeps rules parsed so far), bad declarations skipped
- HTML: unclosed tags auto-close, unmatched close tags ignored, empty input → default root
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

`StyleWritten(u64)` bitmask tracks explicitly-set properties. 14 inheritable property bits:
`INHERIT_COLOR`, `INHERIT_FONT_SIZE`, `INHERIT_FONT_WEIGHT`, `INHERIT_LINE_HEIGHT`,
`INHERIT_TEXT_ALIGN`, `INHERIT_WHITE_SPACE`, `INHERIT_VISIBILITY`, `INHERIT_CURSOR`,
`INHERIT_LETTER_SPACING`, `INHERIT_WORD_SPACING`, `INHERIT_TEXT_TRANSFORM`,
`INHERIT_TEXT_INDENT`, `INHERIT_WORD_BREAK`, `INHERIT_DIRECTION`.

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
- **final_h respects avail_h**: When a node has no explicit height, its final height is `max(content + padding + border, avail_h)`. This propagates cross-axis stretch from the parent through `layout_node` without needing an extra parameter.
- **Main-axis intrinsic sizing**: `Tree::intrinsic_width()` recursively measures text-extent / child-sum / child-max
- **Flex-grow**: distributes remaining main-axis space after intrinsic sizing
- **Flex-shrink**: proportional shrinking only on _definite_ main axes (main_budget > 0), respecting min constraints
- **Dimension::Calc resolution**: `calc(100% - 20px)` resolved during layout with actual parent sizes
- The parent determines child size, not the child's own direction

### Visual Comparison Testing

Pixel-accurate comparison against Chrome headless reference:

- **Chrome headless**: `google-chrome-stable --headless=new --screenshot=<path> --window-size=800,600 --force-device-scale-factor=1 <url>` → exact 800×600 PNG, zero decorations
- **Engine capture**: `maim -u -i <wid>` → window content; may include CSD title bar (37px on current system), crop with `convert -crop 800x600+0+37`
- **Known text offset**: Engine uses `chars × font_size × 0.55` width and `lines × font_size × line_height` height estimation. This produces ~4px vertical cumulative offset vs real font metrics. 88% exact pixel match is the current baseline.
- **Film-strip transition testing**: Freeze animations at N time steps using `animation-play-state: paused` + negative `animation-delay` in CSS, then screenshot all frames in one image. No JS/Puppeteer needed.

### Hit Test & Event Dispatch

- `Tree::hit_test(pos)` — depth-first, reverse child order for z-order
- `Tree::click(pos)` — hit_test + walk parents to find tagged node
- `Tree::dispatch(event)` — full Capture → Target → Bubble propagation
- `Tree::scroll(pos, delta)` — nearest `Overflow::Scroll` container
- `Tree::replay(&scenario)` — scripted interaction replay, zero OS input (see `event.skill.md`)

### Headless GPU & Scenario Runner

- `Gpu::init_headless(w, h)` — no window needed, capture-only GPU renderer
- `Gpu::capture(&mut self, &RenderList) → (w, h, rgba)` — offscreen render to RGBA bytes
- `Gpu::capture_png(&mut self, &RenderList, path)` — capture + save as PNG (BGRA→RGBA auto-converted)
- `Gpu::prepare()` + `Gpu::draw()` — shared helpers, `paint()` and `capture()` are thin wrappers
- Binary `anv-scenario` — parses HTML+CSS, replays a scripted Scenario, saves PNGs at capture points

## Dependencies

- `any-compute-core` — `layout::{Rect, Point, Size}`, `render::{Color, Primitive, RenderList, Border}`,
  `animation::Easing`, `interaction::{InputEvent, EventContext, DispatchResult, Phase}`
- No external deps for the lib — all parsing is hand-rolled
- No feature flags

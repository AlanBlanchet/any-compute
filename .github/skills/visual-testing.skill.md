# Visual Testing Skill — Programmatic Pixel Assertions

All DOM/UI tests MUST verify behavior through rendered pixels and programmatic queries — never trust layout numbers alone. Build a Tree → interact → capture → assert pixels/cursor/tag.

## Test Modes

| Mode | Constructor                                            | Capture                           | Requires         |
| ---- | ------------------------------------------------------ | --------------------------------- | ---------------- |
| GPU  | `TestHarness::from_tree()` / `from_css_html()`         | `capture() → Capture`             | `--features gpu` |
| CPU  | `TestHarness::from_tree_cpu()` / `from_css_html_cpu()` | `capture_cpu(w, h) → PixelBuffer` | nothing          |

CPU tests go in `crates/dom/src/tree_tests.rs`. GPU tests go in `crates/dom/tests/visual.rs`.
Prefer CPU for new behavioral tests — GPU-only for anti-aliasing, shader, or visual regression.

## Imports

```rust
// CPU tests (tree_tests.rs — already inside crate)
use super::*;

// GPU tests (visual.rs — external)
use any_compute_dom::harness::{Capture, TestHarness};
use any_compute_dom::style::*;
use any_compute_dom::tree::*;
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::Color;
```

## Pattern: Build → Interact → Assert

```rust
#[test]
fn button_shows_pointer_cursor() {
    let mut tree = Tree::new(Style::default().w(200.0).h(100.0));
    let btn = tree.add_box(tree.root, Style::default().w(80.0).h(30.0).cursor(Cursor::Pointer));
    tree.tag(btn, "btn");
    tree.layout(Size::new(200.0, 100.0));

    let h = TestHarness::from_tree_cpu(tree, (200, 100));
    assert_eq!(h.cursor_at((40.0, 15.0)), Cursor::Pointer);
    assert_eq!(h.cursor_at((150.0, 80.0)), Cursor::Default);
    assert_eq!(h.tag_at((40.0, 15.0)), Some("btn".into()));
}
```

## Assertion APIs

### TestHarness queries (no rendering needed)

| Method            | Returns          | Use for                                  |
| ----------------- | ---------------- | ---------------------------------------- |
| `tag_at(pos)`     | `Option<String>` | Hit-test: correct element receives click |
| `style_at(pos)`   | `Option<Style>`  | Verify computed style at position        |
| `cursor_at(pos)`  | `Cursor`         | Verify cursor icon (Pointer, Text, etc.) |
| `is_hovered(pos)` | `bool`           | Verify hover state after `hover()`       |

### Capture (GPU) pixel assertions

| Method                                | Use for                         |
| ------------------------------------- | ------------------------------- |
| `pixel(x, y) → Color`                 | Exact pixel color               |
| `region_uniform(x,y,w,h, color, tol)` | Area is one solid color         |
| `count_color(x,y,w,h, target, tol)`   | Count matching pixels in region |
| `diff_count(&other, tol)`             | State-change visual diff        |
| `save_png(path)`                      | Human inspection output         |

### PixelBuffer (CPU) pixel assertions

| Method                          | Use for                      |
| ------------------------------- | ---------------------------- |
| `pixel(x, y) → Color`           | Exact pixel color            |
| `diff(&other, tol) → u32`       | Pixel diff count             |
| `diff_ratio(&other, tol) → f64` | Fraction of differing pixels |

## What to Test

### Cursor behavior

- Elements with `cursor: pointer` CSS return `Cursor::Pointer` from `cursor_at()`
- Nested elements inherit cursor from ancestors when not set
- Overlapping elements: topmost z-order determines cursor

### Click routing

- `tag_at(pos)` returns the correct tag for overlapping/nested elements
- `tree.click(pos)` returns the expected tag

### Scroll behavior

- `h.scroll(pos, delta)` clips content: pixels outside container are clear color
- Scroll thumbs appear when content overflows (`Overflow::Scroll` style)

### Hover state

- `h.hover(pos)` → `h.is_hovered(pos)` is true
- Hover on child doesn't leak to sibling

### Pixel rendering (GPU/CPU)

- Background colors render at expected positions
- Borders render with correct width/color
- Rounded corners: corner pixels are clear, center pixels are fill

## Anti-Patterns (DO NOT)

- Don't assert exact layout coordinates — assert pixel colors or tag hits instead
- Don't use `#[ignore]` — fix or delete broken tests
- Don't skip CPU tests just because GPU exists — CPU tests run everywhere
- Don't assert on floating-point layout values with `==` — use tolerance
- Don't write tests that only check `tree.slot(id).rect` without rendering

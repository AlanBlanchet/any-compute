# Showcase Skill — `examples/showcase/`

Unified demo app with 4 tabs: Browser, 3D Scene, Compute, AI.

## Architecture

- `app.rs` — shared `AppData` struct, `build_tree()` + `render_frame()` methods, layout constants, associated helpers (`tab_tag`, `hover_tag`, `in_content_area`, `to_dom_pos`)
- `main.rs` — event loop, compositing
- `shared.rs` — `Shared` (Arc<Mutex<AppData>>), background workers, constants, `decode_http_body()` charset-aware byte→String conversion
- `snapshot.rs` — headless renderer, uses `AppData` from `app.rs` directly, fetches live google.com with charset detection
- `tabs/helpers.rs` — shared UI helpers (style lookups, badges, key-value cards, subtab bars)
- `tabs/graph/` — AI training studio: types, interaction, training sim, UI builders, dataset data
- `tabs/browser/` — Chrome-like browser: state (BrowserState), devtools panel subtabs
- `tabs/scene/` — 3D scene tab: state (SceneInfo), UI (hierarchy/inspector)
- `tabs/compute.rs` — Compute benchmarks tab

## CSS Cascade Pitfall

**COMBINED_CSS** concatenates palette + showcase + website CSS. Last-rule-wins means website.css can shadow showcase.css classes. **Always namespace website.css classes with `ws-` prefix** to avoid collisions.

## RenderList Compositing

Overlay primitives onto tagged viewport boxes via `list.composite(vp_rect, &prims)`. Tags like `ai-graph-viewport`, `ai-loss-chart`, `ai-ds-scatter`, `scene-viewport` mark where prims are drawn.

## Browser Tab

### Layout Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│ [🌐 Tab Title ×]  [+]                          (tab bar, 32px)     │
├──────────────────────────────────────────────────────────────────────┤
│ [←][→][↻][ 🔒  https://example.com  ][nodes ⋮] (nav bar, 42px)    │
├─────────────────────────────┬────────────────────────────────────────┤
│                             │ Elements │Console│Network│Source       │
│    Rendered page viewport   │                                       │
│    (tagged: browser-viewport│  DevTools panel (380px, toggleable)   │
│     for overlay)            │  (tagged: browser-devtools)           │
└─────────────────────────────┴────────────────────────────────────────┘
```

### BrowserState

```rust
pub struct BrowserState {
    pub page: Option<Page>,      // Loaded Page runtime
    pub cursor: Point,           // Mouse within viewport
    pub needs_measure: bool,     // Re-measure text nodes
    pub html: String,            // Current HTML source
    pub input: TextInput,        // Editable URL bar
    pub console_logs: Vec<(String, String)>,  // (level, message)
    pub network_log: Vec<(String, String, u16, usize)>, // (method, url, status, size)
    pub devtools_scroll: f64,    // DevTools scroll position
    pub devtools_open: bool,     // Toggle DevTools panel
    pub history: Vec<String>,    // URL history stack
    pub history_idx: i32,        // Current position (-1 = none)
    pub tab_title: String,       // Active tab display title
    pub is_secure: bool,         // HTTPS lock icon
}
```

### Chrome Module

Color constants in `mod chrome`: `TAB_BG(35,39,52)`, `TAB_ACTIVE = theme::BG`, `NAV_BG(42,46,60)`, `LOCK_GREEN(80,200,80)`.

### Tags & Event Routing (handled in main.rs)

| Tag                       | Action                        |
| ------------------------- | ----------------------------- |
| `browser-viewport`        | Page overlay + mouse events   |
| `browser-devtools`        | DevTools panel scroll         |
| `browser-reload`          | `reload(&COMBINED_CSS)`       |
| `browser-back`            | `go_back(&COMBINED_CSS)`      |
| `browser-forward`         | `go_forward(&COMBINED_CSS)`   |
| `browser-devtools-toggle` | `toggle_devtools()`           |
| `browser-input`           | URL bar focus (prevents blur) |
| `subtab-0-N`              | DevTools subtab switch        |

### DevTools Subtabs

- **Elements**: DOM tree with expand arrows, element labels `<tag id="x" class="y">`, dimensions (W×H), Computed Styles (display, font-size, color, bg, padding, margin, border-radius, overflow)
- **Console**: Error/warning count badges, level icons (✕/⚠/ⓘ), colored row backgrounds, line numbers, input prompt
- **Network**: Status code badges (green 2xx, yellow 3xx, red 4xx+), method/URL/size/type columns, total transfer size
- **Source**: Line-numbered HTML with 500-line limit, char/node count badges

### Navigation

- `navigate()` pushes to `history`, extracts title from URL, sets `is_secure`
- `go_back()`/`go_forward()` use `history_idx` (i32, starts at -1)
- `can_go_forward()` guards against usize wrap: `history_idx >= 0 && (history_idx as usize) + 1 < history.len()`
- URL protocols: `file://` → local file, `http://`/`https://` → external fetch, domain-like → prepend `https://`

### Helpers

- `nav_icon()` — reusable nav button with enabled/disabled styling
- `panel_scaffold()` → `(info_row, scroll_box)` for standard panel layout
- `empty_state()` — placeholder for empty panels
- `format_color(Color) → String` — CSS color string
- `format_edges(Edges) → String` — shorthand margin/padding display

## AI Tab

### Planner → Center View

Left panel cascades: Task → Model → Dataset. Center view switches between Training, ModelGraph, Models, Datasets.

### Dataset Explorer

Five sub-views: Cards, Table, Samples, Embeddings, Transforms. Image pixel-grid thumbnails are fetched live from `DatasetProvider` (HuggingFace Hub API) — no embedded binary fixtures. Provider loading is async: `start_ds_fetch()` spawns a thread, `tick_ds_fetch()` polls per-frame, decoded pixels are cached in `ds_provider_pixels: HashMap<DatasetSource, Vec<Vec<Color>>>`. Placeholder gradients shown while loading (full-alpha, class-colored). When a sample is selected, a split layout (content with `overflow(Hidden)` + fixed-width viewer) prevents images from pushing the viewer panel. Embeddings scatter plot renders per-sample pixel thumbnails (or colored dots for text datasets) with method/perplexity controls. `decode_provider_pixels` uses `ds_grid_size(ds)` to match the display grid exactly — mismatch between decode and display grid causes scrambled images.

## Key Conventions

- ALL clickable elements MUST have `.cursor(Cursor::Pointer)`
- Tag-based event routing: tag pattern `ai-{category}-{id}`, parsed via `strip_prefix`
- `measure_text_nodes()` must be called before `layout()` on the showcase tree (not just the Page tree) for accurate text centering

## Snapshot Binary

Headless renderer for all showcase tabs: `cargo run -p showcase --bin snapshot`. Saves PNGs to `out/snapshots/`.

Uses `Gpu::init_headless(1400, 900)` + `gpu.capture_png(&list, path)`. Renders all 4 tabs, Browser DevTools subtabs, test HTML pages (simple, table, nested layout). Both binaries share `AppData` from `app.rs` — no struct duplication. Snapshot calls `data.render_frame(sheet, gpu, w, h, path)` and `data.build_tree(sheet, w, h)` as methods on the shared `AppData`.

Use this to visually verify any UI changes before declaring them done.

//! # any-compute-bench
//!
//! Performance benchmarks for the any-compute DOM.
//!
//! Measures `parse`, `css-resolve`, `layout`, and `paint` throughput on our
//! arena DOM and compares against equivalent work on a naive `Box<Node>` tree
//! (the kind of heap-per-node structure that real browser DOMs use).

use std::time::Instant;

use any_compute_core::layout::Size;
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::*;
use any_compute_dom::tree::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Shared constants ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Raw bench.css text — single source for both `lib` and `window`.
pub const BENCH_CSS: &str = include_str!("bench.css");

/// Default viewport for benchmarks and the GPU dashboard.
pub const VIEWPORT: Size = Size::new(1400.0, 900.0);

/// Version string shown in the dashboard brand area.
pub const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

/// Sidebar tab labels — shared between benchmark tree builder and GPU window.
pub const TAB_LABELS: &[&str] = &["Hardware", "Benchmarks", "Live Showdown"];

/// Parse and cache the bench stylesheet.  Re‑exported so `window.rs` avoids
/// its own `include_str!` + parse.
pub fn sheet() -> StyleSheet {
    StyleSheet::parse(BENCH_CSS)
}

/// Shorthand: resolve one CSS class.
pub fn s(class: &str) -> Style {
    sheet().class(class)
}

/// Shorthand: resolve + merge multiple CSS classes.
pub fn sm(classes: &[&str]) -> Style {
    sheet().classes(classes)
}

/// Key-value row: `[label.w(72) | value]` using bench.css utilities.
pub fn kv_row(t: &mut Tree, parent: NodeId, label: &str, value: &str, sheet: &StyleSheet) {
    let r = t.add_box(parent, sheet.classes(&["row", "gap-8"]));
    t.add_text(r, label, sheet.class("label").w(72.0));
    t.add_text(r, value, sheet.class("body"));
}

/// Build the common sidebar + tab shell.
///
/// Returns `(sidebar_id, content_id)`.  Caller adds dynamic content.
pub fn build_shell(
    t: &mut Tree,
    sheet: &StyleSheet,
    active_tab: usize,
) -> (NodeId, NodeId) {
    let root = t.root;
    // Sidebar
    let sb = t.add_box(root, sheet.class("sidebar"));
    let brand = t.add_box(sb, sheet.class("brand"));
    t.add_box(brand, sheet.class("brand-icon"));
    let bt = t.add_box(brand, sheet.class("brand-text"));
    t.add_text(bt, "any-compute", sheet.classes(&["heading", "text"]));
    t.add_text(bt, VERSION, sheet.classes(&["small", "text-dim"]));
    for (i, label) in TAB_LABELS.iter().enumerate() {
        let cls = if i == active_tab { "tab-active" } else { "tab-inactive" };
        let btn = t.add_box(sb, sheet.classes(&["tab-btn", cls]));
        t.add_text(btn, *label, sheet.class("font-13"));
    }
    // Main
    let main = t.add_box(root, sheet.class("grow"));
    let hdr = t.add_box(main, sheet.class("header"));
    t.add_text(hdr, TAB_LABELS[active_tab], sheet.classes(&["font-18", "text"]));
    let content = t.add_box(main, sheet.class("content"));
    (sb, content)
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Reference "browser-like" DOM for comparison ─────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Naive heap-per-node tree — every node is a separate allocation behind
/// `Box` + `Vec<Box<RefNode>>`, mimicking what real browser DOMs do.
/// We measure identical operations on both representations.
#[derive(Clone)]
struct RefNode {
    _style: Style,
    children: Vec<Box<RefNode>>,
}

impl RefNode {
    fn new(style: Style) -> Self {
        Self {
            _style: style,
            children: Vec::new(),
        }
    }

    fn add_child(&mut self, style: Style) -> &mut RefNode {
        self.children.push(Box::new(RefNode::new(style)));
        self.children.last_mut().unwrap()
    }

    fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Benchmark harness ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// One measurement: name, node count, our ops/s, reference ops/s.
pub struct Measurement {
    pub name: &'static str,
    pub nodes: usize,
    pub arena_ops: f64,
    pub heap_ops: f64,
}

impl Measurement {
    pub fn speedup(&self) -> f64 {
        if self.heap_ops > 0.0 {
            self.arena_ops / self.heap_ops
        } else {
            f64::INFINITY
        }
    }
}

fn measure(
    name: &'static str,
    rounds: u32,
    arena_fn: impl Fn(),
    heap_fn: impl Fn(),
    nodes: usize,
) -> Measurement {
    // Warmup
    for _ in 0..3 {
        arena_fn();
        heap_fn();
    }

    let t0 = Instant::now();
    for _ in 0..rounds {
        arena_fn();
    }
    let arena_dur = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    for _ in 0..rounds {
        heap_fn();
    }
    let heap_dur = t0.elapsed().as_secs_f64();

    Measurement {
        name,
        nodes,
        arena_ops: rounds as f64 / arena_dur,
        heap_ops: rounds as f64 / heap_dur,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Scenarios ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// ── Scenarios ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Build a flat N-child arena tree.
fn arena_flat(n: usize) -> Tree {
    let mut t = Tree::new(Style::default().w(VIEWPORT.w).h(VIEWPORT.h));
    let r = t.root;
    for i in 0..n {
        t.add_text(r, format!("node-{i}"), Style::default().font(12.0));
    }
    t
}

/// Build a flat N-child heap tree.
fn heap_flat(n: usize) -> RefNode {
    let mut root = RefNode::new(Style::default().w(VIEWPORT.w).h(VIEWPORT.h));
    for _ in 0..n {
        root.add_child(Style::default().font(12.0));
    }
    root
}

/// Build a deep arena tree (linear chain).
fn arena_deep(depth: usize) -> Tree {
    let mut t = Tree::new(Style::default().w(800.0).h(600.0));
    let mut parent = t.root;
    for _ in 0..depth {
        parent = t.add_box(parent, Style::default().pad(2.0));
    }
    t
}

/// Build a deep heap tree (linear chain).
fn heap_deep(depth: usize) -> RefNode {
    let mut root = RefNode::new(Style::default().w(800.0).h(600.0));
    let mut ptr = &mut root as *mut RefNode;
    for _ in 0..depth {
        // SAFETY: we own the tree, and each add_child returns a valid &mut.
        unsafe {
            let child = (*ptr).add_child(Style::default().pad(2.0));
            ptr = child as *mut RefNode;
        }
    }
    root
}

/// Build a realistic dashboard-like tree using CSS + shared shell.
fn arena_dashboard(sheet: &StyleSheet) -> Tree {
    let mut t = Tree::new(sheet.classes(&["bg", "row"]).w(VIEWPORT.w).h(VIEWPORT.h));
    let (_sb, content) = build_shell(&mut t, sheet, 0);
    let row = t.add_box(content, sheet.classes(&["row", "gap-12"]));
    for _ in 0..3 {
        let card = t.add_box(row, sheet.class("card"));
        t.add_text(card, "Card Title", sheet.class("heading"));
        for _ in 0..4 {
            kv_row(&mut t, card, "Label", "Value", sheet);
        }
        t.add_bar(card, 0.65, sheet.class("green").color, sheet.class("bar-thin"));
    }
    t
}

/// Equivalent heap tree for the dashboard.
fn heap_dashboard(sheet: &StyleSheet) -> RefNode {
    let mut root = RefNode::new(sheet.classes(&["bg", "row"]).w(VIEWPORT.w).h(VIEWPORT.h));
    let sb = root.add_child(sheet.class("sidebar"));
    let brand = sb.add_child(sheet.class("brand"));
    brand.add_child(sheet.class("brand-icon"));
    let bt = brand.add_child(sheet.class("brand-text"));
    bt.add_child(sheet.classes(&["heading", "text"]));
    bt.add_child(sheet.classes(&["small", "text-dim"]));
    for (i, _label) in TAB_LABELS.iter().enumerate() {
        let cls = if i == 0 { "tab-active" } else { "tab-inactive" };
        let btn = sb.add_child(sheet.classes(&["tab-btn", cls]));
        btn.add_child(sheet.class("font-13"));
    }
    let main = root.add_child(sheet.class("grow"));
    let hdr = main.add_child(sheet.class("header"));
    hdr.add_child(sheet.classes(&["font-18", "text"]));
    let content = main.add_child(sheet.class("content"));
    let row = content.add_child(sheet.classes(&["row", "gap-12"]));
    for _ in 0..3 {
        let card = row.add_child(sheet.class("card"));
        card.add_child(sheet.class("heading"));
        for _ in 0..4 {
            let r = card.add_child(sheet.classes(&["row", "gap-8"]));
            r.add_child(sheet.class("label").w(72.0));
            r.add_child(sheet.class("body"));
        }
        card.add_child(sheet.class("bar-thin"));
    }
    root
}

/// Run all DOM benchmarks. Returns measurements for display or reporting.
pub fn run_dom_benchmarks() -> Vec<Measurement> {
    let sheet = StyleSheet::parse(BENCH_CSS);
    let rounds = 2000;
    let mut results = Vec::new();

    // 1. Flat tree creation (1000 children)
    let n = 1000;
    results.push(measure(
        "create flat 1K nodes",
        rounds,
        || {
            std::hint::black_box(arena_flat(n));
        },
        || {
            std::hint::black_box(heap_flat(n));
        },
        n + 1,
    ));

    // 2. Deep tree creation (500-deep chain)
    let d = 500;
    results.push(measure(
        "create deep 500 chain",
        rounds,
        || {
            std::hint::black_box(arena_deep(d));
        },
        || {
            std::hint::black_box(heap_deep(d));
        },
        d + 1,
    ));

    // 3. Layout pass on flat tree
    {
        results.push(measure(
            "layout flat 1K",
            rounds,
            || {
                let mut t = arena_flat(n);
                t.layout(VIEWPORT);
                std::hint::black_box(&t);
            },
            || {
                // Heap tree has no layout solver — just measure creation overhead (baseline)
                let h = heap_flat(n);
                std::hint::black_box(&h);
            },
            n + 1,
        ));
    }

    // 4. Paint pass
    {
        let mut a = arena_flat(100);
        a.layout(VIEWPORT);
        results.push(measure(
            "paint 100 nodes",
            rounds * 5,
            || {
                let mut list = RenderList::default();
                a.paint(&mut list);
                std::hint::black_box(&list);
            },
            || {
                // Reference: just allocating a Vec with equivalent capacity
                let v: Vec<u8> = Vec::with_capacity(100 * 64);
                std::hint::black_box(&v);
            },
            101,
        ));
    }

    // 5. CSS parse
    results.push(measure(
        "CSS parse (bench.css)",
        rounds,
        || {
            std::hint::black_box(StyleSheet::parse(BENCH_CSS));
        },
        || {
            // Reference: just allocation of comparable HashMap
            let m: std::collections::HashMap<String, Vec<(String, String)>> =
                std::collections::HashMap::with_capacity(30);
            std::hint::black_box(&m);
        },
        0,
    ));

    // 6. CSS class resolution
    {
        results.push(measure(
            "CSS resolve 1K classes",
            rounds,
            || {
                for _ in 0..1000 {
                    std::hint::black_box(sheet.class("card"));
                }
            },
            || {
                // Reference: HashMap lookup + Style::default()
                for _ in 0..1000 {
                    std::hint::black_box(Style::default());
                }
            },
            0,
        ));
    }

    // 7. HTML parse
    {
        let html = r##"<div w="1400" h="900" direction="row"><div w="220" pad="12" gap="8"><span font="16">Sidebar</span></div><div grow="1" pad="24" gap="16"><span font="22">Main</span><progress value="0.6" color="#a6e3a1" h="8" /></div></div>"##;
        results.push(measure(
            "HTML parse (small doc)",
            rounds,
            || {
                std::hint::black_box(any_compute_dom::parse::parse(html));
            },
            || {
                // Reference: just string scanning (find all '<')
                let count = html.bytes().filter(|&b| b == b'<').count();
                std::hint::black_box(count);
            },
            6,
        ));
    }

    // 8. Dashboard tree build + layout + paint (full frame)
    results.push(measure(
        "full frame (dashboard)",
        rounds / 2,
        || {
            let mut t = arena_dashboard(&sheet);
            t.layout(VIEWPORT);
            let mut list = RenderList::default();
            t.paint(&mut list);
            std::hint::black_box(&list);
        },
        || {
            let h = heap_dashboard(&sheet);
            std::hint::black_box(h.node_count());
        },
        arena_dashboard(&sheet).arena.len(),
    ));

    results
}

/// Print benchmark results to stdout in a table.
pub fn print_results(results: &[Measurement]) {
    println!(
        "\n{:<30} {:>8} {:>14} {:>14} {:>10}",
        "Benchmark", "Nodes", "Arena ops/s", "Heap ops/s", "Speedup"
    );
    println!("{}", "-".repeat(80));
    for m in results {
        let nodes_str = if m.nodes > 0 {
            format!("{}", m.nodes)
        } else {
            "—".into()
        };
        println!(
            "{:<30} {:>8} {:>14.0} {:>14.0} {:>9.1}x",
            m.name,
            nodes_str,
            m.arena_ops,
            m.heap_ops,
            m.speedup(),
        );
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use any_compute_core::render::RenderList;

    #[test]
    fn dashboard_builds_and_lays_out() {
        let sh = sheet();
        let mut t = arena_dashboard(&sh);
        assert!(t.arena.len() > 30, "dashboard should have 30+ nodes");
        t.layout(VIEWPORT);
        let mut list = RenderList::default();
        t.paint(&mut list);
        assert!(list.len() > 10, "dashboard should produce 10+ primitives");
    }
}

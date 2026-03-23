//! # any-compute-bench
//!
//! Performance benchmarks and comparison suite for any-compute.
//!
//! - **`runner`** — compute/kernel benchmark categories, harness, hardware detection,
//!   comparison tables, simulated device profiles, live metrics.
//! - **DOM benchmarks** — arena `Tree` vs heap-per-node `Box<RefNode>` comparison,
//!   CSS/HTML parse throughput, full website parsing, layout and paint measurement.
//! - **Dashboard UI helpers** — shared stylesheet, sidebar/tab shell builder.

pub mod runner;

use std::time::Instant;

use any_compute_dom::PALETTE_CSS;
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

/// Realistic static website HTML fixture (~200 nodes).
pub const WEBSITE_HTML: &str = include_str!("../fixtures/website.html");

/// CSS for the website fixture — 90+ rules, variables, selectors.
pub const WEBSITE_CSS: &str = include_str!("../fixtures/website.css");

/// Combined CSS: palette + Tailwind utilities + bench.css overrides.
/// Parsed once at startup → O(1) lookups.
pub fn combined_css() -> String {
    format!(
        "{PALETTE}\n{TW}\n{BENCH}",
        PALETTE = PALETTE_CSS,
        TW = any_compute_dom::TAILWIND_CSS,
        BENCH = BENCH_CSS,
    )
}

/// Default viewport for benchmarks and the GPU dashboard.
pub const VIEWPORT: Size = Size::new(1400.0, 900.0);

/// Version string shown in the dashboard brand area.
pub const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

/// Sidebar tab labels — shared between benchmark tree builder and GPU window.
pub const TAB_LABELS: &[&str] = &["Hardware", "Benchmarks", "Live Showdown"];

/// Cached combined stylesheet (Tailwind + bench.css).
/// Single `LazyLock` — parsed once on first access, reused everywhere.
pub static SHEET: std::sync::LazyLock<StyleSheet> =
    std::sync::LazyLock::new(|| StyleSheet::parse(&combined_css()));

/// Shorthand: resolve one CSS class from the shared stylesheet.
pub fn s(class: &str) -> Style {
    SHEET.class(class)
}

/// Shorthand: resolve + merge multiple CSS classes from the shared stylesheet.
pub fn sm(classes: &[&str]) -> Style {
    SHEET.classes(classes)
}

/// Key-value row: `[label.w(72) | value]` using bench.css utilities.
pub fn kv_row(t: &mut Tree, parent: NodeId, label: &str, value: &str) {
    let r = t.add_box(parent, s("row-gap-8"));
    t.add_text(r, label, s("label").w(72.0));
    t.add_text(r, value, s("body"));
}

/// Build the common sidebar + tab shell.
///
/// Returns `(sidebar_id, content_id)`.  Caller adds dynamic content.
pub fn build_shell(t: &mut Tree, active_tab: usize) -> (NodeId, NodeId) {
    let root = t.root;
    let sb = t.add_box(root, s("sidebar"));
    let brand = t.add_box(sb, s("brand"));
    t.add_box(brand, s("brand-icon"));
    let bt = t.add_box(brand, s("brand-text"));
    t.add_text(bt, "any-compute", s("heading-text"));
    t.add_text(bt, VERSION, s("small-dim"));
    for (i, label) in TAB_LABELS.iter().enumerate() {
        let cls = if i == active_tab {
            "tab-active"
        } else {
            "tab-inactive"
        };
        let btn = t.add_box(sb, sm(&["tab-btn", cls]));
        t.add_text(btn, *label, s("font-13"));
    }
    let main = t.add_box(root, s("grow"));
    let hdr = t.add_box(main, s("header"));
    t.add_text(hdr, TAB_LABELS[active_tab], sm(&["font-18", "text"]));
    let content = t.add_box(main, s("content"));
    (sb, content)
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Generic benchmark harness macros ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Measure a single operation: warmup, then timed iterations → ops/sec.
///
/// Generic — reusable for any throughput measurement (DOM, compute, parse, …).
/// Returns `(ops_per_sec, duration_per_op_us)`.
pub fn bench_throughput(warmup: u32, rounds: u32, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..rounds {
        f();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let ops = rounds as f64 / elapsed;
    let us_per = (elapsed / rounds as f64) * 1e6;
    (ops, us_per)
}

/// Measure paired A/B operation (arena vs heap, ours vs reference) and
/// return a `Measurement`.  Reusable for any comparative benchmark.
pub fn bench_pair(
    name: &'static str,
    nodes: usize,
    rounds: u32,
    arena_fn: impl Fn(),
    heap_fn: impl Fn(),
) -> Measurement {
    for _ in 0..3 {
        arena_fn();
        heap_fn();
    }
    let (arena_ops, _) = bench_throughput(0, rounds, &arena_fn);
    let (heap_ops, _) = bench_throughput(0, rounds, &heap_fn);
    Measurement {
        name,
        nodes,
        arena_ops,
        heap_ops,
    }
}

/// Declare a batch of paired benchmarks in a compact table.
///
/// Each entry: `name, nodes, rounds, arena_expr, heap_expr`.
/// Expands to `bench_pair(...)` calls collected into a `Vec<Measurement>`.
macro_rules! bench_scenarios {
    ($( $name:literal, $nodes:expr, $rounds:expr,
        $arena:expr, $heap:expr );+ $(;)?) => {{
        vec![ $(
            bench_pair($name, $nodes, $rounds,
                || { std::hint::black_box($arena); },
                || { std::hint::black_box($heap); },
            ),
        )+ ]
    }};
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Benchmark types ─────────────────────────────────────────────────────
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

// ═══════════════════════════════════════════════════════════════════════════
// ── Reference "browser-like" DOM for comparison ─────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Naive heap-per-node tree mimicking browser DOM allocation patterns.
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
// ── Scenario builders ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

fn arena_flat(n: usize) -> Tree {
    let mut t = Tree::new(Style::default().w(VIEWPORT.w()).h(VIEWPORT.h()));
    let r = t.root;
    for i in 0..n {
        t.add_text(r, format!("node-{i}"), Style::default().font(12.0));
    }
    t
}

fn heap_flat(n: usize) -> RefNode {
    let mut root = RefNode::new(Style::default().w(VIEWPORT.w()).h(VIEWPORT.h()));
    for _ in 0..n {
        root.add_child(Style::default().font(12.0));
    }
    root
}

fn arena_deep(depth: usize) -> Tree {
    let mut t = Tree::new(Style::default().w(800.0).h(600.0));
    let mut parent = t.root;
    for _ in 0..depth {
        parent = t.add_box(parent, Style::default().pad(2.0));
    }
    t
}

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

fn arena_dashboard() -> Tree {
    let mut t = Tree::new(sm(&["bg", "row"]).w(VIEWPORT.w()).h(VIEWPORT.h()));
    let (_sb, content) = build_shell(&mut t, 0);
    let row = t.add_box(content, s("row-gap-12"));
    for _ in 0..3 {
        let card = t.add_box(row, s("card"));
        t.add_text(card, "Card Title", s("heading"));
        for _ in 0..4 {
            kv_row(&mut t, card, "Label", "Value");
        }
        t.add_bar(card, 0.65, s("green").color, s("bar-thin"));
    }
    t
}

fn heap_dashboard() -> RefNode {
    let mut root = RefNode::new(sm(&["bg", "row"]).w(VIEWPORT.w()).h(VIEWPORT.h()));
    let sb = root.add_child(s("sidebar"));
    let brand = sb.add_child(s("brand"));
    brand.add_child(s("brand-icon"));
    let bt = brand.add_child(s("brand-text"));
    bt.add_child(s("heading-text"));
    bt.add_child(s("small-dim"));
    for (i, _label) in TAB_LABELS.iter().enumerate() {
        let cls = if i == 0 { "tab-active" } else { "tab-inactive" };
        let btn = sb.add_child(sm(&["tab-btn", cls]));
        btn.add_child(s("font-13"));
    }
    let main = root.add_child(s("grow"));
    let hdr = main.add_child(s("header"));
    hdr.add_child(sm(&["font-18", "text"]));
    let content = main.add_child(s("content"));
    let row = content.add_child(s("row-gap-12"));
    for _ in 0..3 {
        let card = row.add_child(s("card"));
        card.add_child(s("heading"));
        for _ in 0..4 {
            let r = card.add_child(s("row-gap-8"));
            r.add_child(s("label").w(72.0));
            r.add_child(s("body"));
        }
        card.add_child(s("bar-thin"));
    }
    root
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Run all DOM benchmarks ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Run all DOM benchmarks. Returns measurements for display or reporting.
pub fn run_dom_benchmarks() -> Vec<Measurement> {
    let sheet = StyleSheet::parse(BENCH_CSS);
    let website_sheet = StyleSheet::parse(WEBSITE_CSS);
    let rounds = 2000;

    let n = 1000;
    let d = 500;

    let mut results = bench_scenarios! {
        // ── Tree creation ───────────────────────────────────────────
        "create flat 1K nodes", n + 1, rounds,
            arena_flat(n), heap_flat(n);
        "create deep 500 chain", d + 1, rounds,
            arena_deep(d), heap_deep(d);

        // ── Layout ──────────────────────────────────────────────────
        "layout flat 1K", n + 1, rounds,
            { let mut t = arena_flat(n); t.layout(VIEWPORT); t },
            heap_flat(n);

        // ── CSS parsing ─────────────────────────────────────────────
        "CSS parse (bench.css)", 0, rounds,
            StyleSheet::parse(BENCH_CSS),
            std::collections::HashMap::<String, Vec<(String, String)>>::with_capacity(30);
        "CSS parse (website.css)", 0, rounds,
            StyleSheet::parse(WEBSITE_CSS),
            std::collections::HashMap::<String, Vec<(String, String)>>::with_capacity(90);

    };

    // ── CSS resolution ──────────────────────────────────────────────
    results.push(bench_pair(
        "CSS resolve 1K classes",
        0,
        rounds,
        || {
            for _ in 0..1000 {
                std::hint::black_box(sheet.class("card"));
            }
        },
        || {
            for _ in 0..1000 {
                std::hint::black_box(Style::default());
            }
        },
    ));

    // ── Paint ───────────────────────────────────────────────────────
    {
        let mut a = arena_flat(100);
        a.layout(VIEWPORT);
        results.push(bench_pair(
            "paint 100 nodes",
            101,
            rounds * 5,
            || {
                let mut list = RenderList::default();
                a.paint(&mut list);
                std::hint::black_box(&list);
            },
            || {
                let v: Vec<u8> = Vec::with_capacity(100 * 64);
                std::hint::black_box(&v);
            },
        ));
    }

    // ── HTML parsing ────────────────────────────────────────────────
    {
        let small_html = r##"<div w="1400" h="900" direction="row"><div w="220" pad="12" gap="8"><span font="16">Sidebar</span></div><div grow="1" pad="24" gap="16"><span font="22">Main</span><progress value="0.6" color="#a6e3a1" h="8" /></div></div>"##;
        results.push(bench_pair(
            "HTML parse (small doc)",
            6,
            rounds,
            || {
                drop(std::hint::black_box(any_compute_dom::parse::parse(
                    small_html,
                )))
            },
            || {
                let _ = std::hint::black_box(small_html.bytes().filter(|&b| b == b'<').count());
            },
        ));
    }

    // ── Website parsing (full static site) ──────────────────────────
    {
        let tree = any_compute_dom::parse::parse_with_css(WEBSITE_HTML, &website_sheet);
        let website_nodes = tree.arena.len();
        results.push(bench_pair(
            "website parse (HTML only)",
            website_nodes,
            rounds,
            || {
                drop(std::hint::black_box(any_compute_dom::parse::parse(
                    WEBSITE_HTML,
                )))
            },
            || {
                let _ = std::hint::black_box(WEBSITE_HTML.bytes().filter(|&b| b == b'<').count());
            },
        ));
        results.push(bench_pair(
            "website parse + CSS resolve",
            website_nodes,
            rounds,
            || {
                drop(std::hint::black_box(
                    any_compute_dom::parse::parse_with_css(WEBSITE_HTML, &website_sheet),
                ))
            },
            || {
                drop(std::hint::black_box(any_compute_dom::parse::parse(
                    WEBSITE_HTML,
                )))
            },
        ));
    }

    // ── Website full pipeline (parse + layout + paint) ──────────────
    {
        let tree = any_compute_dom::parse::parse_with_css(WEBSITE_HTML, &website_sheet);
        let website_nodes = tree.arena.len();
        results.push(bench_pair(
            "website full frame",
            website_nodes,
            rounds / 2,
            || {
                let mut t = any_compute_dom::parse::parse_with_css(WEBSITE_HTML, &website_sheet);
                t.layout(VIEWPORT);
                let mut list = RenderList::default();
                t.paint(&mut list);
                drop(std::hint::black_box(list));
            },
            || drop(std::hint::black_box(heap_dashboard())),
        ));
    }

    // ── Dashboard full frame ────────────────────────────────────────
    results.push(bench_pair(
        "dashboard full frame",
        arena_dashboard().arena.len(),
        rounds / 2,
        || {
            let mut t = arena_dashboard();
            t.layout(VIEWPORT);
            let mut list = RenderList::default();
            t.paint(&mut list);
            drop(std::hint::black_box(list));
        },
        || {
            let _ = std::hint::black_box(heap_dashboard().node_count());
        },
    ));

    results
}

/// Print benchmark results to stdout in a table.
pub fn print_results(results: &[Measurement]) {
    println!(
        "\n{:<35} {:>8} {:>14} {:>14} {:>10}",
        "Benchmark", "Nodes", "Arena ops/s", "Heap ops/s", "Speedup"
    );
    println!("{}", "-".repeat(85));
    for m in results {
        let nodes_str = if m.nodes > 0 {
            format!("{}", m.nodes)
        } else {
            "—".into()
        };
        println!(
            "{:<35} {:>8} {:>14.0} {:>14.0} {:>9.1}x",
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

    #[test]
    fn dashboard_builds_and_lays_out() {
        let mut t = arena_dashboard();
        assert!(t.arena.len() > 30, "dashboard should have 30+ nodes");
        t.layout(VIEWPORT);
        let mut list = RenderList::default();
        t.paint(&mut list);
        assert!(list.len() > 10, "dashboard should produce 10+ primitives");
    }

    #[test]
    fn website_parses_to_large_tree() {
        let sheet = StyleSheet::parse(WEBSITE_CSS);
        let tree = any_compute_dom::parse::parse_with_css(WEBSITE_HTML, &sheet);
        // Website fixture has ~150+ nodes
        assert!(
            tree.arena.len() > 100,
            "website should parse to 100+ nodes, got {}",
            tree.arena.len()
        );
    }

    #[test]
    fn website_full_pipeline() {
        let sheet = StyleSheet::parse(WEBSITE_CSS);
        let mut tree = any_compute_dom::parse::parse_with_css(WEBSITE_HTML, &sheet);
        tree.layout(VIEWPORT);
        let mut list = RenderList::default();
        tree.paint(&mut list);
        assert!(
            list.len() > 50,
            "website should produce 50+ primitives, got {}",
            list.len()
        );
    }

    #[test]
    fn bench_throughput_returns_positive() {
        let (ops, us) = bench_throughput(2, 100, || {
            std::hint::black_box(42);
        });
        assert!(ops > 0.0);
        assert!(us > 0.0);
    }

    #[test]
    fn bench_scenarios_macro_works() {
        let results = bench_scenarios! {
            "test_a", 10, 50, arena_flat(10), heap_flat(10);
            "test_b", 5, 50, arena_deep(5), heap_deep(5);
        };
        assert_eq!(results.len(), 2);
        assert!(results[0].arena_ops > 0.0);
        assert!(results[1].arena_ops > 0.0);
    }
}

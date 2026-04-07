//! Page runtime — loads an HTML document with embedded CSS and JavaScript.
//!
//! A [`Page`] is the top-level container that owns a DOM [`Tree`], a CSS
//! [`StyleSheet`], and a JS [`Vm`].  It connects them:
//!
//! 1. Parse HTML → extract `<style>` and `<script>` blocks
//! 2. Parse extracted CSS → `StyleSheet`
//! 3. Parse cleaned HTML with the stylesheet → `Tree`
//! 4. Install DOM API native functions into the `Vm`
//! 5. Execute `<script>` blocks in document order
//!
//! After construction the `Page` can be laid out, painted, and receive
//! events exactly like a bare `Tree` — plus it can re-evaluate JS on
//! demand (e.g. event handlers, timers).
//!
//! ## Memory model
//!
//! The `Tree` lives as host data inside the `Vm` (via `Vm::set_host`).
//! Native DOM functions access it through `Vm::host_mut::<PageHost>()`.
//! This avoids `Rc<RefCell<>>` overhead and keeps ownership linear:
//!
//! ```text
//! Page ──owns──► Vm ──host──► PageHost { tree, sheet }
//! ```

use std::sync::Arc;

use any_compute_core::layout::Size;
use any_compute_core::render::RenderList;
use any_compute_js::{JsError, JsValue, Vm};

use super::css::StyleSheet;
use super::parse::{extract_resources, parse_with_css};
use super::style::Style;
use super::tree::{NodeId, Tree};

// ═══════════════════════════════════════════════════════════════════════════
// ── Host data stored inside the VM ──────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Data attached to the JS VM as host state, accessible from native functions.
pub struct PageHost {
    pub tree: Tree,
    pub sheet: Arc<StyleSheet>,
}

// ═══════════════════════════════════════════════════════════════════════════
// ── DOM API installation ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Install browser-like DOM API on a `Vm` whose host is `PageHost`.
///
/// Provides a `document` global with `getElementById`, `querySelector`,
/// and `createElement`.
///
/// Element handles expose `get_textContent`, `set_textContent`,
/// `setStyle`, `getStyle`, `appendChild`, `classListAdd`, `classListRemove`.
///
/// Element handles are `JsObject`s with an `__id` property storing the
/// `NodeId` index. Native methods use `require_host` / `require_host_mut`
/// to access the `Tree` through the VM's host data.
fn install_dom_api(vm: &mut Vm) {
    use any_compute_js::value::JsObject;
    let mut doc = JsObject::new();
    doc.set(
        "getElementById".into(),
        JsValue::Object(JsObject::native_fn(dom_get_element_by_id)),
    );
    doc.set(
        "querySelector".into(),
        JsValue::Object(JsObject::native_fn(dom_query_selector)),
    );
    doc.set(
        "createElement".into(),
        JsValue::Object(JsObject::native_fn(dom_create_element)),
    );
    vm.define_global("document", JsValue::Object(doc));
}

// ── Native DOM helpers ──────────────────────────────────────────────────

/// Extract `NodeId` from an element handle's `__id` property.
fn node_id_from(el: &JsValue) -> Option<NodeId> {
    if let JsValue::Object(obj) = el {
        if let JsValue::Number(n) = obj.get("__id") {
            return Some(NodeId(n as usize));
        }
    }
    None
}

/// Extract `NodeId` from `this`, returning a JS error if missing.
fn require_node(this: &JsValue) -> Result<NodeId, JsError> {
    node_id_from(this).ok_or_else(|| JsError::Runtime("not an element".into()))
}

/// Borrow the `PageHost` from the VM (read-only).
fn require_host(vm: &Vm) -> Result<&PageHost, JsError> {
    vm.host::<PageHost>()
        .ok_or_else(|| JsError::Internal("no page host".into()))
}

/// Borrow the `PageHost` from the VM (mutable).
fn require_host_mut(vm: &mut Vm) -> Result<&mut PageHost, JsError> {
    vm.host_mut::<PageHost>()
        .ok_or_else(|| JsError::Internal("no page host".into()))
}

/// Build a JS element handle wrapping a `NodeId`.
fn element_handle(id: NodeId) -> JsValue {
    use any_compute_js::value::JsObject;
    let mut obj = JsObject::new();
    obj.set("__id".into(), JsValue::Number(id.0 as f64));
    // Attach method bridges so JS can call el.textContent, el.style, etc.
    obj.set(
        "get_textContent".into(),
        JsValue::Object(JsObject::native_fn(dom_get_text_content)),
    );
    obj.set(
        "set_textContent".into(),
        JsValue::Object(JsObject::native_fn(dom_set_text_content)),
    );
    obj.set(
        "setStyle".into(),
        JsValue::Object(JsObject::native_fn(dom_set_style)),
    );
    obj.set(
        "getStyle".into(),
        JsValue::Object(JsObject::native_fn(dom_get_style)),
    );
    obj.set(
        "appendChild".into(),
        JsValue::Object(JsObject::native_fn(dom_append_child)),
    );
    obj.set(
        "classListAdd".into(),
        JsValue::Object(JsObject::native_fn(dom_classlist_add)),
    );
    obj.set(
        "classListRemove".into(),
        JsValue::Object(JsObject::native_fn(dom_classlist_remove)),
    );
    JsValue::Object(obj)
}

fn dom_get_element_by_id(
    vm: &mut Vm,
    _this: &JsValue,
    args: &[JsValue],
) -> Result<JsValue, JsError> {
    let id_str = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host(vm)?;
    for (i, slot) in host.tree.arena.iter().enumerate() {
        if slot.id.as_deref() == Some(&id_str) {
            return Ok(element_handle(NodeId(i)));
        }
    }
    Ok(JsValue::Null)
}

fn dom_query_selector(vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let sel = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host(vm)?;

    // Simple selector matching: .class, #id, tag
    for (i, slot) in host.tree.arena.iter().enumerate() {
        let matched = if let Some(cls) = sel.strip_prefix('.') {
            slot.class_list.iter().any(|c| c == cls)
        } else if let Some(id) = sel.strip_prefix('#') {
            slot.id.as_deref() == Some(id)
        } else {
            slot.element == sel
        };
        if matched {
            return Ok(element_handle(NodeId(i)));
        }
    }
    Ok(JsValue::Null)
}

fn dom_create_element(vm: &mut Vm, _this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let tag = args
        .first()
        .map(|a| a.to_js_string())
        .unwrap_or_else(|| "div".into());
    let host = require_host_mut(vm)?;
    // Add as child of root with default style.
    let id = host.tree.add_box(host.tree.root, Style::default());
    host.tree.slot_mut(id).element = tag;
    Ok(element_handle(id))
}

fn dom_get_text_content(
    vm: &mut Vm,
    this: &JsValue,
    _args: &[JsValue],
) -> Result<JsValue, JsError> {
    let nid = require_node(this)?;
    let host = require_host(vm)?;
    if nid.0 >= host.tree.arena.len() {
        return Ok(JsValue::Undefined);
    }
    let slot = host.tree.slot(nid);
    match &slot.kind {
        super::tree::NodeKind::Text(s) => Ok(JsValue::String(s.clone())),
        _ => {
            let mut buf = String::new();
            collect_text(&host.tree, nid, &mut buf);
            Ok(JsValue::String(buf))
        }
    }
}

/// Recursively collect text content from a subtree.
fn collect_text(tree: &Tree, id: NodeId, buf: &mut String) {
    let slot = tree.slot(id);
    if let super::tree::NodeKind::Text(s) = &slot.kind {
        buf.push_str(s);
    }
    for &child in &slot.children.clone() {
        collect_text(tree, child, buf);
    }
}

fn dom_set_text_content(vm: &mut Vm, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let nid = require_node(this)?;
    let text = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host_mut(vm)?;
    if nid.0 >= host.tree.arena.len() {
        return Ok(JsValue::Undefined);
    }
    // If slot is Text, update directly. Otherwise, clear children and set.
    let slot = host.tree.slot_mut(nid);
    match &mut slot.kind {
        super::tree::NodeKind::Text(s) => {
            *s = text;
        }
        _ => {
            // Clear children and add a text child.
            slot.children.clear();
            host.tree.add_text(nid, &text, Style::default());
        }
    }
    host.tree.needs_layout = true;
    host.tree.needs_paint = true;
    Ok(JsValue::Undefined)
}

fn dom_set_style(vm: &mut Vm, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let nid = require_node(this)?;
    let prop = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let val = args.get(1).map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host_mut(vm)?;
    if nid.0 >= host.tree.arena.len() {
        return Ok(JsValue::Undefined);
    }
    // Use the existing compile_attr pipeline to set style.
    if let Some(op) = super::parse::compile_attr(&prop, &val) {
        op.apply(&mut host.tree.slot_mut(nid).style);
        host.tree.needs_layout = true;
        host.tree.needs_paint = true;
    }
    Ok(JsValue::Undefined)
}

fn dom_get_style(vm: &mut Vm, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let nid = require_node(this)?;
    let prop = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host(vm)?;
    if nid.0 >= host.tree.arena.len() {
        return Ok(JsValue::Undefined);
    }
    let style = &host.tree.slot(nid).style;
    // Return property values for the most common properties.
    let val = match prop.as_str() {
        "display" => style.display.to_css().to_string(),
        "width" => dim_to_string(style.width),
        "height" => dim_to_string(style.height),
        "color" => format!("{:?}", style.color),
        "background" | "background-color" | "backgroundColor" => {
            format!("{:?}", style.background)
        }
        "opacity" => style.opacity.to_string(),
        "font-size" | "fontSize" => format!("{}px", style.font_size),
        _ => String::new(),
    };
    Ok(JsValue::String(val))
}

fn dim_to_string(d: super::style::Dimension) -> String {
    match d {
        super::style::Dimension::Auto => "auto".into(),
        super::style::Dimension::Px(v) => format!("{v}px"),
        super::style::Dimension::Percent(v) => format!("{v}%"),
        super::style::Dimension::Calc { percent, px } => {
            format!("calc({percent}% + {px}px)")
        }
    }
}

fn dom_append_child(vm: &mut Vm, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let parent_id = require_node(this)?;
    let child_id = args
        .first()
        .and_then(node_id_from)
        .ok_or_else(|| JsError::Runtime("child is not an element".into()))?;
    let host = require_host_mut(vm)?;
    // Re-parent the child node.
    if parent_id.0 < host.tree.arena.len() && child_id.0 < host.tree.arena.len() {
        // Remove from old parent's children list.
        if let Some(old_parent) = host.tree.slot(child_id).parent {
            let old_children = &mut host.tree.slot_mut(old_parent).children;
            old_children.retain(|c| *c != child_id);
        }
        host.tree.slot_mut(child_id).parent = Some(parent_id);
        host.tree.slot_mut(parent_id).children.push(child_id);
        host.tree.needs_layout = true;
        host.tree.needs_paint = true;
    }
    Ok(JsValue::Undefined)
}

fn dom_classlist_add(vm: &mut Vm, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let nid = require_node(this)?;
    let cls = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host_mut(vm)?;
    if nid.0 < host.tree.arena.len() && !cls.is_empty() {
        let slot = host.tree.slot_mut(nid);
        if !slot.class_list.contains(&cls) {
            slot.class_list.push(cls);
        }
    }
    Ok(JsValue::Undefined)
}

fn dom_classlist_remove(vm: &mut Vm, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsError> {
    let nid = require_node(this)?;
    let cls = args.first().map(|a| a.to_js_string()).unwrap_or_default();
    let host = require_host_mut(vm)?;
    if nid.0 < host.tree.arena.len() {
        let slot = host.tree.slot_mut(nid);
        slot.class_list.retain(|c| c != &cls);
    }
    Ok(JsValue::Undefined)
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Page ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A loaded web page — owns the DOM tree, stylesheet, and JS runtime.
///
/// Constructed from raw HTML (with embedded `<style>` and `<script>`),
/// or from separate HTML/CSS/JS strings.
///
/// After loading, exposes the same layout/paint/event API as `Tree`,
/// plus `eval()` for running additional JS.
pub struct Page {
    pub vm: Vm,
    /// Script execution errors (non-fatal — logged, not propagated).
    pub errors: Vec<JsError>,
}

impl Page {
    /// Load a page from a single HTML document.
    ///
    /// Extracts `<style>` and `<script>` blocks, parses CSS, builds the
    /// DOM tree, installs the DOM API, and executes scripts.
    pub fn load(html: &str) -> Self {
        Self::load_with(html, "", &[])
    }

    /// Load a page from HTML + external CSS + external JS sources.
    ///
    /// The external CSS is prepended before any `<style>` content.
    /// External scripts run before inline `<script>` blocks.
    pub fn load_with(html: &str, external_css: &str, external_scripts: &[&str]) -> Self {
        let (clean_html, resources) = extract_resources(html);

        // Merge CSS: external first, then inline <style> blocks.
        let mut full_css = String::new();
        if !external_css.is_empty() {
            full_css.push_str(external_css);
            full_css.push('\n');
        }
        if !resources.css.is_empty() {
            full_css.push_str(&resources.css);
        }

        let sheet = StyleSheet::parse(&full_css);
        let tree = parse_with_css(&clean_html, &sheet);
        let sheet = Arc::new(sheet);

        // Set up the VM with DOM API.
        let mut vm = any_compute_js::create_vm();
        install_dom_api(&mut vm);
        vm.set_host(PageHost {
            tree,
            sheet: sheet.clone(),
        });

        // Execute external scripts first, then inline.
        let mut errors = Vec::new();
        for src in external_scripts {
            if let Err(e) = vm.eval(src) {
                log::warn!("JS error (external): {e}");
                errors.push(e);
            }
        }
        for src in &resources.scripts {
            if let Err(e) = vm.eval(src) {
                log::warn!("JS error (inline): {e}");
                errors.push(e);
            }
        }

        Page { vm, errors }
    }

    // ── Tree delegation ─────────────────────────────────────────────

    /// Access the DOM tree (read-only).
    pub fn tree(&self) -> &Tree {
        &self.vm.host::<PageHost>().unwrap().tree
    }

    /// Access the DOM tree (mutable).
    pub fn tree_mut(&mut self) -> &mut Tree {
        &mut self.vm.host_mut::<PageHost>().unwrap().tree
    }

    /// Access the stylesheet.
    pub fn sheet(&self) -> &Arc<StyleSheet> {
        &self.vm.host::<PageHost>().unwrap().sheet
    }

    /// Evaluate additional JavaScript in the page context.
    pub fn eval(&mut self, source: &str) -> Result<JsValue, JsError> {
        self.vm.eval(source)
    }

    /// Run flexbox layout.
    pub fn layout(&mut self, viewport: Size) {
        self.tree_mut().layout(viewport);
    }

    /// Emit render primitives into a render list.
    pub fn paint(&self, list: &mut RenderList) {
        self.tree().paint(list);
    }

    /// Clear dirty flags after painting.
    pub fn post_paint(&mut self) {
        self.tree_mut().post_paint();
    }

    /// Whether the tree needs re-layout.
    pub fn needs_layout(&self) -> bool {
        self.tree().needs_layout
    }

    /// Whether the tree needs re-paint.
    pub fn needs_paint(&self) -> bool {
        self.tree().needs_paint()
    }

    /// Advance animations/transitions.
    pub fn tick(&mut self, dt: f64) -> super::tree::TickResult {
        self.tree_mut().tick(dt)
    }

    /// Dispatch an input event.
    pub fn dispatch(
        &mut self,
        event: any_compute_core::interaction::InputEvent,
    ) -> any_compute_core::interaction::DispatchResult {
        self.tree_mut().dispatch(event)
    }

    /// Start CSS animations.
    pub fn start_animations(&mut self) {
        self.tree_mut().start_animations();
    }

    /// Measure text nodes with a real font measurer.
    pub fn measure_text_nodes(&mut self, f: impl FnMut(&str, f64) -> f64) {
        self.tree_mut().measure_text_nodes(f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_minimal_page() {
        let page = Page::load("<div>hello</div>");
        assert!(page.errors.is_empty());
        assert!(page.tree().arena.len() >= 1);
    }

    #[test]
    fn extract_style_and_script() {
        let html = r#"
            <div id="app">
                <style>.red { color: red; }</style>
                <p>Hello</p>
                <script>
                    let el = document.getElementById("app");
                </script>
            </div>
        "#;
        let (clean, res) = extract_resources(html);
        assert!(clean.contains("<p>Hello</p>"));
        assert!(!clean.contains("<style>"));
        assert!(!clean.contains("<script>"));
        assert!(res.css.contains(".red"));
        assert_eq!(res.scripts.len(), 1);
        assert!(res.scripts[0].contains("getElementById"));
    }

    #[test]
    fn js_modifies_dom() {
        let html = r#"
            <div>
                <p id="target">old</p>
                <script>
                    let el = document.getElementById("target");
                    el.set_textContent("new");
                </script>
            </div>
        "#;
        let page = Page::load(html);
        assert!(page.errors.is_empty(), "JS errors: {:?}", page.errors);

        // Find the text node and verify it was changed.
        let tree = page.tree();
        let mut found = false;
        for slot in &tree.arena {
            if let super::super::tree::NodeKind::Text(ref s) = slot.kind {
                if s == "new" {
                    found = true;
                    break;
                }
            }
        }
        assert!(found, "JS should have set textContent to 'new'");
    }

    #[test]
    fn inline_css_applies() {
        let html = r#"
            <div>
                <style>.big { font-size: 32px; }</style>
                <p class="big">Large</p>
            </div>
        "#;
        let page = Page::load(html);
        // Find the "big" node and check font size.
        let tree = page.tree();
        let big_node = tree
            .arena
            .iter()
            .find(|s| s.class_list.contains(&"big".to_string()));
        assert!(big_node.is_some(), "should find node with class 'big'");
        let font_size = big_node.unwrap().style.font_size;
        assert!(
            (font_size - 32.0).abs() < 0.1,
            "font_size should be 32, got {font_size}"
        );
    }

    #[test]
    fn page_layout_and_paint() {
        let html = r#"
            <div style="width: 200px; height: 100px;">
                <p>Hello World</p>
            </div>
        "#;
        let mut page = Page::load(html);
        page.layout(Size::new(800.0, 600.0));
        let mut list = RenderList {
            primitives: Vec::new(),
        };
        page.paint(&mut list);
        assert!(
            !list.primitives.is_empty(),
            "should produce render primitives"
        );
    }

    #[test]
    fn multiple_scripts_execute_in_order() {
        let html = r#"
            <div>
                <script>var x = 1;</script>
                <script>x = x + 1;</script>
                <script>
                    let el = document.querySelector("div");
                </script>
            </div>
        "#;
        let page = Page::load(html);
        assert!(page.errors.is_empty(), "JS errors: {:?}", page.errors);
    }

    /// Combined CSS for the showcase website (palette + showcase + website).
    fn website_css() -> String {
        format!(
            "{}\n{}\n{}",
            crate::PALETTE_CSS,
            include_str!("../../../examples/showcase/showcase.css"),
            include_str!("../../../examples/showcase/website.css"),
        )
    }

    const WEBSITE_HTML: &str = include_str!("../../../examples/showcase/website.html");

    /// Load the showcase website page with full CSS stack.
    fn load_website() -> Page {
        let css = website_css();
        let mut page = Page::load_with(WEBSITE_HTML, &css, &[]);
        page.layout(Size::new(800.0, 600.0));
        page
    }

    #[test]
    fn website_nav_text_vertically_centered() {
        let page = load_website();
        let tree = page.tree();

        // Find all nav-item nodes and verify text is vertically centered.
        for (i, slot) in tree.arena.iter().enumerate() {
            if !slot.class_list.iter().any(|c| c == "nav-item") {
                continue;
            }
            let parent_rect = slot.rect;
            // Find the text child
            for &child_id in &slot.children {
                let child = &tree.arena[child_id.0];
                if matches!(child.kind, super::super::tree::NodeKind::Text(_)) {
                    let child_rect = child.rect;
                    // Text should be roughly vertically centered in the 36px nav-item.
                    // The center of the text should be within 4px of the center of the parent.
                    let parent_center_y = parent_rect.y() + parent_rect.h() / 2.0;
                    let child_center_y = child_rect.y() + child_rect.h() / 2.0;
                    let delta = (parent_center_y - child_center_y).abs();
                    assert!(
                        delta < 6.0,
                        "nav-item[{i}] text not centered: parent_center_y={parent_center_y:.1}, \
                         child_center_y={child_center_y:.1}, delta={delta:.1}"
                    );
                }
            }
        }
    }

    #[test]
    fn website_cards_within_viewport() {
        let page = load_website();
        let tree = page.tree();

        // Every demo-card should fit within the root's width (800px viewport).
        let root_right = 800.0;
        for (i, slot) in tree.arena.iter().enumerate() {
            if !slot.class_list.iter().any(|c| c == "demo-card") {
                continue;
            }
            let card_right = slot.rect.x() + slot.rect.w();
            assert!(
                card_right <= root_right + 1.0,
                "demo-card[{i}] overflows viewport: right={card_right:.1} > {root_right}"
            );
        }
    }

    #[test]
    fn website_loads_without_panic() {
        let css = website_css();
        // Simulate browser reload — should not panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Page::load_with(WEBSITE_HTML, &css, &[])
        }));
        assert!(result.is_ok(), "Page load panicked");
    }

    #[test]
    fn empty_html_loads_without_panic() {
        let css = website_css();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Page::load_with("", &css, &[])
        }));
        assert!(result.is_ok(), "Empty HTML load panicked");
    }

    #[test]
    fn reload_with_various_html_no_panic() {
        let css = website_css();
        // Simulate what happens when the URL bar changes:
        // the browser reloads with the same or different HTML.
        let cases = [
            WEBSITE_HTML,
            "",
            "<div>hello</div>",
            "<h1>Error</h1><p>Page not found</p>",
            "<div class=\"root\"><div class=\"ws-sidebar\"></div></div>",
        ];
        for html in cases {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut page = Page::load_with(html, &css, &[]);
                page.layout(Size::new(800.0, 600.0));
                let mut list = RenderList {
                    primitives: Vec::new(),
                };
                page.paint(&mut list);
            }));
            assert!(
                result.is_ok(),
                "Load+layout+paint panicked for html len={}",
                html.len()
            );
        }
    }

    #[test]
    fn website_layout_performance() {
        let css = website_css();
        let mut page = Page::load_with(WEBSITE_HTML, &css, &[]);
        // Warm up
        page.layout(Size::new(800.0, 600.0));

        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            page.tree_mut().needs_layout = true;
            page.layout(Size::new(800.0, 600.0));
        }
        let elapsed = start.elapsed();
        let per_layout_us = elapsed.as_micros() as f64 / iterations as f64;
        eprintln!(
            "website layout: {per_layout_us:.0}µs/call ({:.0} FPS budget at 60Hz = {:.0}µs)",
            1_000_000.0 / per_layout_us,
            16_666.0
        );
        // Layout should complete in under 2ms (even in debug mode) to maintain 60fps
        assert!(
            per_layout_us < 2000.0,
            "layout too slow: {per_layout_us:.0}µs (must be < 2000µs)"
        );
    }

    #[test]
    fn deeply_nested_html_no_crash() {
        // Google.com can produce deeply nested HTML (500+ levels).
        // Layout and paint must not stack-overflow.
        let depth = 500;
        let html = "<div>".repeat(depth) + "hello" + &"</div>".repeat(depth);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut page = Page::load(&html);
            page.layout(Size::new(800.0, 600.0));
            let mut list = RenderList::default();
            page.paint(&mut list);
        }));
        assert!(result.is_ok(), "deeply nested HTML should not crash");
    }

    #[test]
    fn large_flat_html_no_crash() {
        // Many sibling nodes (mimics minified HTML with thousands of elements).
        let mut html = String::from("<div>");
        for i in 0..2000 {
            html.push_str(&format!("<span>item {i}</span>"));
        }
        html.push_str("</div>");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut page = Page::load(&html);
            page.layout(Size::new(800.0, 600.0));
            let mut list = RenderList::default();
            page.paint(&mut list);
        }));
        assert!(result.is_ok(), "large flat HTML should not crash");
    }

    #[test]
    fn google_like_html_no_crash() {
        // Load saved google HTML (run: curl -sL https://www.google.com > /tmp/google_test.html).
        let Ok(raw) = std::fs::read("/tmp/google_test.html") else {
            eprintln!("skipping: /tmp/google_test.html not found");
            return;
        };
        let html = String::from_utf8_lossy(&raw).into_owned();
        let mut page = Page::load(&html);
        page.layout(Size::new(800.0, 600.0));
        let mut list = RenderList::default();
        page.paint(&mut list);
        assert!(
            page.tree().arena.len() > 0,
            "google.com should produce nodes"
        );
    }
}

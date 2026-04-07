//! Browser tab — chrome-like browser with tab bar, nav bar, viewport, and
//! collapsible DevTools panel.
//!
//! Layout (DevTools open):
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ [Tab1]  [+]                                                        │
//! ├──────────────────────────────────────────────────────────────────────┤
//! │ [←][→][↻][ 🔒  https://example.com          ][⋮]                   │
//! ├─────────────────────────────────┬────────────────────────────────────┤
//! │                                 │ Elements │Console│Network│Source  │
//! │    Rendered page preview        │                                   │
//! │                                 │  DevTools panel content           │
//! └─────────────────────────────────┴────────────────────────────────────┘
//!
//! The Page overlay is rendered in main.rs after layout, clipped to the
//! "browser-viewport" node's rect (no hardcoded offsets).

use any_compute_core::interaction::TextInput;
use any_compute_core::layout::Point;
use any_compute_core::render::Color;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::page::Page;
use any_compute_dom::style::*;
use any_compute_dom::theme;
use any_compute_dom::tree::*;

use super::helpers::{ERROR_PAGE_HTML, badge, build_subtab_bar, format_human, s, sm};

/// DevTools tab labels.

mod state;
mod devtools;

pub use state::*;
use devtools::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tab building
// ═══════════════════════════════════════════════════════════════════════════

/// Build browser tab: tab bar + nav bar + viewport + optional DevTools.
pub fn build(
    sheet: &StyleSheet,
    t: &mut Tree,
    parent: NodeId,
    state: &BrowserState,
    subtab: usize,
) {
    // Outer column: tab bar, nav bar, content area
    let outer = t.add_box(parent, Style::default().grow(1.0));

    // ── Tab bar (Chrome-like) ───────────────────────────────
    build_tab_bar(sheet, t, outer, state);

    // ── Navigation bar ──────────────────────────────────────
    build_nav(sheet, t, outer, state);

    // ── Content area: viewport + optional DevTools ──────────
    let content_row = t.add_box(outer, Style::default().row().grow(1.0));

    // Main viewport
    let viewport_box = t.add_box(
        content_row,
        Style::default()
            .grow(1.0)
            .bg(theme::BG)
            .overflow(Overflow::Hidden),
    );
    let viewport = t.add_box(viewport_box, Style::default().grow(1.0));
    t.tag(viewport, VIEWPORT_TAG);

    // DevTools panel (right side, togglable)
    if state.devtools_open {
        // Separator
        t.add_box(content_row, Style::default().w(1.0).bg(theme::SURFACE0));

        let devtools = t.add_box(
            content_row,
            Style::default()
                .w(380.0)
                .min_w(300.0)
                .bg(theme::SIDEBAR_BG)
                .gap(0.0)
                .overflow(Overflow::Scroll),
        );
        t.tag(devtools, DEVTOOLS_TAG);
        t.slot_mut(devtools).scroll.y = state.devtools_scroll;
        build_subtab_bar(sheet, t, devtools, DEVTABS, subtab, "subtab-0-");
        match subtab {
            0 => build_elements(sheet, t, devtools, state),
            1 => build_console(sheet, t, devtools, state),
            2 => build_network(sheet, t, devtools, state),
            3 => build_source(sheet, t, devtools, state),
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab bar — Chrome-like with tab + new tab + close
// ═══════════════════════════════════════════════════════════════════════════

fn build_tab_bar(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, state: &BrowserState) {
    let bar = t.add_box(
        parent,
        Style::default()
            .row()
            .h(32.0)
            .bg(chrome::TAB_BG)
            .align(Align::End)
            .gap(1.0)
            .pad_xy(8.0, 0.0),
    );

    // Active tab
    let tab = t.add_box(
        bar,
        Style::default()
            .row()
            .h(28.0)
            .w(200.0)
            .min_w(100.0)
            .pad_xy(12.0, 0.0)
            .gap(6.0)
            .bg(chrome::TAB_ACTIVE)
            .radius(6.0)
            .align(Align::Center),
    );
    // Tab icon (globe)
    t.add_text(tab, "🌐", s(sheet, "font-11"));
    // Tab title
    let title: String = state.tab_title.chars().take(20).collect();
    t.add_text(
        tab,
        &title,
        s(sheet, "font-11").color(theme::TEXT).grow(1.0),
    );
    // Tab close button
    let close = t.add_box(
        tab,
        Style::default()
            .wh(16.0, 16.0)
            .align(Align::Center)
            .justify(Justify::Center)
            .radius(3.0)
            .cursor(Cursor::Pointer),
    );
    t.add_text(close, "×", s(sheet, "font-11").color(theme::OVERLAY0));

    // New tab button
    let new_tab = t.add_box(
        bar,
        Style::default()
            .wh(28.0, 28.0)
            .align(Align::Center)
            .justify(Justify::Center)
            .radius(6.0)
            .cursor(Cursor::Pointer),
    );
    t.add_text(new_tab, "+", s(sheet, "font-14").color(theme::OVERLAY0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Navigation bar — back, forward, reload, lock, URL, devtools toggle
// ═══════════════════════════════════════════════════════════════════════════

/// Small icon button in the nav bar.
fn nav_icon(
    sheet: &StyleSheet,
    t: &mut Tree,
    parent: NodeId,
    icon: &str,
    tag: &str,
    enabled: bool,
) {
    let color = if enabled {
        theme::SUBTEXT0
    } else {
        Color::rgba(100, 100, 110, 80)
    };
    let btn = t.add_box(
        parent,
        Style::default()
            .wh(28.0, 28.0)
            .align(Align::Center)
            .justify(Justify::Center)
            .radius(6.0)
            .cursor(if enabled {
                Cursor::Pointer
            } else {
                Cursor::Default
            }),
    );
    t.tag(btn, tag);
    t.add_text(btn, icon, s(sheet, "font-14").color(color));
}

fn build_nav(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, state: &BrowserState) {
    let bar = t.add_box(
        parent,
        Style::default()
            .row()
            .h(NAV_H)
            .gap(4.0)
            .pad_xy(8.0, 0.0)
            .align(Align::Center)
            .bg(chrome::NAV_BG),
    );

    // ── Back / Forward / Reload ─────────────────────────────
    nav_icon(sheet, t, bar, "←", "browser-back", state.can_go_back());
    nav_icon(
        sheet,
        t,
        bar,
        "→",
        "browser-forward",
        state.can_go_forward(),
    );
    nav_icon(sheet, t, bar, "↻", "browser-reload", true);

    // ── Address bar with lock icon ──────────────────────────
    let addr_wrap = t.add_box(
        bar,
        Style::default()
            .row()
            .grow(1.0)
            .h(30.0)
            .bg(theme::BG)
            .radius(15.0)
            .pad_xy(10.0, 0.0)
            .align(Align::Center)
            .gap(6.0),
    );

    // Security lock icon
    if state.is_secure {
        t.add_text(
            addr_wrap,
            "🔒",
            s(sheet, "font-11").color(chrome::LOCK_GREEN),
        );
    } else {
        t.add_text(addr_wrap, "ⓘ", s(sheet, "font-11").color(theme::OVERLAY0));
    }

    // Address bar text input
    t.build_address_bar(addr_wrap, &state.input, &AddressBarStyle::default());

    // ── Right side: page info + devtools toggle ─────────────
    let nodes = state.node_count();
    t.add_text(bar, &format!("{nodes} nodes"), badge(sheet, "green"));

    // DevTools toggle button (F12-like)
    let dt_btn = t.add_box(
        bar,
        Style::default()
            .h(28.0)
            .pad_xy(8.0, 0.0)
            .align(Align::Center)
            .justify(Justify::Center)
            .bg(if state.devtools_open {
                theme::ACCENT.with_alpha(40)
            } else {
                theme::SURFACE0
            })
            .radius(6.0)
            .cursor(Cursor::Pointer),
    );
    t.tag(dt_btn, "browser-devtools-toggle");
    t.add_text(
        dt_btn,
        "⋮",
        s(sheet, "font-14").color(if state.devtools_open {
            theme::ACCENT
        } else {
            theme::OVERLAY0
        }),
    );
}

/// Height of the navigation bar (px).
const NAV_H: f64 = 42.0;


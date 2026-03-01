//! Virtualized data grid component — renders only visible rows from a [`DataSource`].
//!
//! All layout math delegates to `any-compute-core::layout::ScrollState::visible_range`.
//! This component is a thin adapter: core owns the logic, dioxus owns the DOM.

use any_compute_core::data::{CellValue, DataSource, VecSource};
use any_compute_core::layout::ScrollState;
use dioxus::prelude::*;
use std::sync::Arc;

const DEFAULT_ROW_HEIGHT: f64 = 28.0;

/// Props for the [`DataGrid`] component.
#[derive(Props, Clone, PartialEq)]
pub struct DataGridProps {
    /// The data source to render.
    #[props(into)]
    pub source: Arc<VecSource>,

    /// Height of each row in pixels.
    #[props(default = DEFAULT_ROW_HEIGHT)]
    pub row_height: f64,

    /// Visible height of the viewport in pixels.
    #[props(default = 600.0)]
    pub viewport_height: f64,
}

/// A virtualized, scrollable data grid.
///
/// Only the rows that fit in the viewport are rendered.
/// Core logic (visible range calculation) lives in `any-compute-core::layout::ScrollState`.
#[component]
pub fn DataGrid(props: DataGridProps) -> Element {
    let scroll = use_signal(ScrollState::default);

    let source = &props.source;
    let row_height = props.row_height;
    let viewport_h = props.viewport_height;

    let scroll_val = scroll.read();
    let range = scroll_val.visible_range(row_height, viewport_h, source.row_count());
    let rows = source.fetch(range.clone());
    let columns = source.columns();
    let total_height = source.row_count() as f64 * row_height;

    rsx! {
        div {
            style: "height: {viewport_h}px; overflow-y: auto; position: relative; font-family: monospace;",
            onscroll: move |evt: Event<ScrollData>| {
                // Placeholder — full scroll integration requires mounted element ref.
                _ = evt;
            },
            // Spacer for correct scrollbar sizing
            div { style: "height: {total_height}px; position: relative;",
                // Header row
                div { style: "display: flex; position: sticky; top: 0; background: #1a1a2e; z-index: 1; border-bottom: 1px solid #333;",
                    for col in columns.iter() {
                        div {
                            style: "flex: 1; padding: 6px 12px; font-weight: bold; color: #e0e0e0; font-size: 13px;",
                            "{col.name}"
                        }
                    }
                }
                // Visible rows only — core computes the range, we just render
                for (i, row) in rows.iter().enumerate() {
                    { let y = (range.start + i) as f64 * row_height;
                      let bg = if (range.start + i) % 2 == 0 { "#16213e" } else { "#0f3460" };
                      rsx! {
                        div {
                            style: "display: flex; position: absolute; top: {y}px; width: 100%; height: {row_height}px; align-items: center; background: {bg}; transition: background 0.15s ease;",
                            for cell in row.iter() {
                                div {
                                    style: "flex: 1; padding: 0 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #ddd; font-size: 13px;",
                                    {format_cell(cell)}
                                }
                            }
                        }
                    }}
                }
            }
        }
    }
}

fn format_cell(v: &CellValue) -> String {
    match v {
        CellValue::Empty => String::new(),
        CellValue::Bool(b) => b.to_string(),
        CellValue::Int(n) => n.to_string(),
        CellValue::Float(f) => format!("{f:.2}"),
        CellValue::Text(s) => s.clone(),
    }
}

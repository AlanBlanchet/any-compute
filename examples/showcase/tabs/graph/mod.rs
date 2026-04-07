//! AI tab — ML training studio.
//!
//! Two-panel layout: Planner (left, 220px) | Tabbed center (Training | Models | Datasets).
//! Center tab bar switches between views. ModelGraph view accessible via
//! "View Architecture" on model cards (shows back button instead of tab bar).

use any_compute_ai::{
    DatasetMeta, DatasetProvider, DatasetSource, HuggingFaceProvider, ImageShape, ModelKind,
    TaskType,
};
use any_compute_core::data::VecSource;
use any_compute_core::render::{Color, RenderList, Renderable};
use any_compute_core::visual::{GraphView, Graphable, VisualGraph};
use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::*;
use any_compute_dom::theme;
use any_compute_dom::tree::*;

use std::collections::HashMap;
use std::sync::mpsc::Receiver;

use super::helpers::{ai_tag, badge, kv_card, s, sm};

/// Tag used to locate the graph viewport for overlay compositing.
pub const VIEWPORT_TAG: &str = "ai-graph-viewport";

mod data;
mod interaction;
mod training;
mod types;
mod ui;

pub(super) use data::*;
pub use types::*;

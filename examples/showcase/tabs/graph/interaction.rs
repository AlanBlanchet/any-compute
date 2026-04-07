use super::*;
impl AiState {
    /// Handle a click tag from the UI. Returns `true` if consumed.
    pub fn handle_click(&mut self, tag: &str) -> bool {
        if let Some(name) = tag.strip_prefix("ai-task-") {
            self.task = TaskType::from_tag(name);
            self.model = None;
            self.dataset = None;
            self.stage = PlannerStage::Model;
            self.center = CenterView::Training;
            return true;
        }
        if let Some(name) = tag.strip_prefix("ai-model-") {
            self.model = ModelKind::from_tag(name);
            self.dataset = None;
            self.stage = PlannerStage::Dataset;
            self.center = CenterView::Training;
            return true;
        }
        if tag == "ai-start-train" {
            self.start_training();
            self.center = CenterView::Training;
            return true;
        }
        if let Some(name) = tag.strip_prefix("ai-view-") {
            if let Some(kind) = ModelKind::from_tag(name) {
                self.open_model_graph(kind);
                self.center = CenterView::ModelGraph;
                return true;
            }
        }
        if tag == "ai-back-train" {
            self.center = CenterView::Training;
            return true;
        }
        // Center view tab switching
        if tag == "ai-center-training" {
            self.center = CenterView::Training;
            return true;
        }
        if tag == "ai-center-models" {
            self.center = CenterView::Models;
            return true;
        }
        if tag == "ai-center-datasets" {
            self.center = CenterView::Datasets;
            return true;
        }
        // Dataset detail view mode switching (must come before ai-ds- planner prefix)
        if tag == "ai-dsview-cards" {
            self.dataset_view = DatasetView::Cards;
            return true;
        }
        if tag == "ai-dsview-table" {
            self.dataset_view = DatasetView::Table;
            return true;
        }
        if tag == "ai-dsview-samples" {
            self.dataset_view = DatasetView::Samples;
            return true;
        }
        if tag == "ai-dsview-embeddings" {
            self.dataset_view = DatasetView::Embeddings;
            return true;
        }
        if tag == "ai-dsview-transforms" {
            self.dataset_view = DatasetView::Transforms;
            return true;
        }
        // Sort column click
        if let Some(col_str) = tag.strip_prefix("ai-ds-sort-") {
            if let Ok(col) = col_str.parse::<usize>() {
                if self.ds_sort_col == col {
                    self.ds_sort_asc = !self.ds_sort_asc;
                } else {
                    self.ds_sort_col = col;
                    self.ds_sort_asc = true;
                }
                return true;
            }
        }
        // Class filter
        if tag == "ai-ds-filter-clear" {
            self.ds_filter_class = None;
            return true;
        }
        if let Some(cls_str) = tag.strip_prefix("ai-ds-filter-") {
            if let Ok(cls) = cls_str.parse::<usize>() {
                self.ds_filter_class = Some(cls);
                return true;
            }
        }
        // Version selector
        if let Some(v_str) = tag.strip_prefix("ai-ds-version-") {
            if let Ok(v) = v_str.parse::<u8>() {
                self.ds_version = v;
                return true;
            }
        }
        // Embedding method selector
        if let Some(m_str) = tag.strip_prefix("ai-ds-embed-method-") {
            if let Ok(m) = m_str.parse::<u8>() {
                self.ds_embed_method = m;
                return true;
            }
        }
        // Embedding perplexity selector
        if let Some(p_str) = tag.strip_prefix("ai-ds-embed-perp-") {
            if let Ok(p) = p_str.parse::<u8>() {
                self.ds_embed_perplexity = p;
                return true;
            }
        }
        // Transform toggle
        if let Some(name) = tag.strip_prefix("ai-ds-tfm-") {
            if let Some(tfm) = Transform::from_tag(name) {
                let bit = 1u8 << (tfm as u8);
                self.ds_transforms ^= bit;
                return true;
            }
        }
        // Sample selection
        if let Some(idx_str) = tag.strip_prefix("ai-ds-sample-") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                // Toggle selection: click again to deselect
                if self.ds_selected_sample == Some(idx) {
                    self.ds_selected_sample = None;
                } else {
                    self.ds_selected_sample = Some(idx);
                }
                return true;
            }
        }
        // Close viewer panel
        if tag == "ai-ds-viewer-close" {
            self.ds_selected_sample = None;
            return true;
        }
        // Back from dataset detail to catalog
        if tag == "ai-ds-back" {
            self.selected_dataset = None;
            self.ds_selected_sample = None;
            self.ds_filter_class = None;
            self.ds_sort_col = 0;
            self.ds_sort_asc = true;
            return true;
        }
        // Dataset card click — select and view details
        if let Some(name) = tag.strip_prefix("ai-dataset-") {
            self.selected_dataset = DatasetSource::from_tag(name);
            self.center = CenterView::Datasets;
            // Trigger provider fetch for image datasets
            if let Some(ds) = self.selected_dataset {
                self.start_ds_fetch(ds);
            }
            return true;
        }
        // Planner dataset selection (must come last — "ai-ds-" is a broad prefix)
        if let Some(name) = tag.strip_prefix("ai-ds-") {
            self.dataset = DatasetSource::from_tag(name);
            self.stage = PlannerStage::Training;
            return true;
        }
        false
    }

    // ── Dataset provider loading ────────────────────────────────────────

    /// Kick off an async fetch for a dataset's pixel data from the provider.
    fn start_ds_fetch(&mut self, ds: DatasetSource) {
        // Already cached or already loading.
        if self.ds_provider_pixels.contains_key(&ds) || self.ds_fetch_rx.is_some() {
            return;
        }
        let ds_id = match ds {
            DatasetSource::Mnist => "mnist",
            DatasetSource::Cifar10 => "cifar10",
            _ => return, // Non-image datasets don't need remote data
        };
        let (tx, rx) = std::sync::mpsc::channel();
        self.ds_fetch_rx = Some((ds, rx));
        let ds_id = ds_id.to_string();
        std::thread::spawn(move || {
            let provider = HuggingFaceProvider::new();
            let _ = tx.send(provider.fetch_samples(&ds_id, 30));
        });
        log::info!("dataset fetch started for {ds:?}");
    }

    /// Poll the background fetch — call every frame.
    pub fn tick_ds_fetch(&mut self) {
        let Some((ds, ref rx)) = self.ds_fetch_rx else {
            return;
        };
        let Ok(result) = rx.try_recv() else {
            return;
        };
        let ds = ds; // copy before borrow
        match result {
            Ok((meta, source)) => {
                let pixels = decode_provider_pixels(ds, &meta, &source);
                self.ds_provider_pixels.insert(ds, pixels);
                self.ds_load_error = None;
                log::info!(
                    "dataset loaded: {} ({} classes)",
                    meta.name,
                    meta.classes.len()
                );
            }
            Err(e) => {
                log::warn!("dataset fetch failed: {e}");
                self.ds_load_error = Some(e);
            }
        }
        self.ds_fetch_rx = None;
    }

    /// Get pixel Colors for a sample's class. Uses provider data if loaded,
    /// otherwise returns a placeholder gradient.
    pub(super) fn pixel_data(&self, ds: DatasetSource, class_idx: usize) -> Vec<Color> {
        let n = ds_grid_size(ds);
        if n == 0 {
            return vec![];
        }
        if let Some(pixels) = self.ds_provider_pixels.get(&ds) {
            if let Some(p) = pixels.get(class_idx) {
                if !p.is_empty() {
                    return p.clone();
                }
            }
        }
        // Placeholder: visible gradient per class while loading
        (0..n * n)
            .map(|px| {
                let frac = px as f64 / (n * n) as f64;
                let base = class_color(class_idx);
                Color::rgba(
                    (base.r as f64 * (0.3 + 0.7 * frac)) as u8,
                    (base.g as f64 * (0.3 + 0.7 * frac)) as u8,
                    (base.b as f64 * (0.3 + 0.7 * frac)) as u8,
                    255,
                )
            })
            .collect()
    }

    /// Open a model's architecture as an interactive graph.
    pub fn open_model_graph(&mut self, kind: ModelKind) {
        self.view = GraphView::default();
        self.cached_visual = Some(CachedGraphInfo {
            visual: kind.to_graph(),
            source: 999,
        });
    }

    /// Configure graph view for an arbitrary `Graphable` source.
    pub fn set_graph(&mut self, visual: VisualGraph<2>, source: usize) {
        self.cached_visual = Some(CachedGraphInfo { visual, source });
        self.view = GraphView::default();
    }

    /// Current cached visual graph (if any).
    pub fn cached_visual(&self) -> Option<&CachedGraphInfo> {
        self.cached_visual.as_ref()
    }

    /// Clear the cached graph (e.g. on tab switch).
    pub fn clear_graph(&mut self) {
        self.cached_visual = None;
        self.view = GraphView::default();
    }

    /// Graph needs rebuild for the given source id.
    pub fn needs_rebuild(&self, source: usize) -> bool {
        self.cached_visual
            .as_ref()
            .map_or(true, |c| c.source != source)
    }

    /// Render the cached graph into `self.prims`.
    pub fn render_graph(&mut self) {
        if let Some(info) = &self.cached_visual {
            let mut prims = RenderList::default();
            info.visual.render(&mut prims, &self.view);
            self.prims = prims;
        }
    }

    /// Pan the graph view by pixel delta (scaled by zoom).
    pub fn on_drag(&mut self, dx: f64, dy: f64) {
        self.view.pan.0[0] += dx / self.view.zoom;
        self.view.pan.0[1] += dy / self.view.zoom;
    }

    /// Zoom the graph at the given local position.
    pub fn on_zoom(&mut self, local: (f64, f64), factor: f64) {
        self.view.zoom_at(
            any_compute_core::layout::V([local.0, local.1]),
            factor,
            0.1,
            10.0,
        );
    }
}

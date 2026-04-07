use super::*;
impl AiState {
    /// Build the full AI tab — planner (left) + tabbed center panel.
    pub fn build(&mut self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let main_row = t.add_box(parent, Style::default().row().grow(1.0).gap(2.0));

        // Left: Planner (task → model → dataset → train)
        self.build_planner(sheet, t, main_row);

        // Center: tabbed panel
        let center = t.add_box(main_row, Style::default().grow(1.0).gap(0.0));

        // Tab bar (only when not viewing model graph — that has its own back button)
        if self.center != CenterView::ModelGraph {
            self.build_center_tabs(sheet, t, center);
        }

        match self.center {
            CenterView::Training => self.build_viewer(sheet, t, center),
            CenterView::ModelGraph => self.build_graph_panel(sheet, t, center),
            CenterView::Models => self.build_models(sheet, t, center),
            CenterView::Datasets => self.build_datasets(sheet, t, center),
        }
    }

    /// Center tab bar.
    fn build_center_tabs(&self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let mut bar_style = Style::default().row().gap(2.0).pad_xy(8.0, 4.0);
        bar_style.flex_shrink = 0.0;
        let bar = t.add_box(parent, bar_style);
        for &(label, tag) in CENTER_TABS {
            let active = match (label, self.center) {
                ("Training", CenterView::Training) => true,
                ("Models", CenterView::Models) => true,
                ("Datasets", CenterView::Datasets) => true,
                _ => false,
            };
            chip_btn(sheet, t, bar, label, active, tag);
        }
        // Runs count badge
        if !self.runs.is_empty() {
            t.add_text(
                bar,
                &format!("{} runs", self.runs.len()),
                badge(sheet, "green"),
            );
        }
    }

    /// Planner panel (left side).
    fn build_planner(&self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(
            parent,
            Style::default()
                .w(220.0)
                .bg(theme::SIDEBAR_BG)
                .pad(12.0)
                .gap(12.0)
                .overflow(Overflow::Scroll),
        );

        // Task
        t.add_text(
            panel,
            "1. Task",
            s(sheet, "font-11").color(if self.stage == PlannerStage::Task {
                theme::ACCENT
            } else {
                theme::OVERLAY0
            }),
        );
        let task_row = t.add_box(panel, s(sheet, "wrap-row"));
        for task in TaskType::ALL {
            let label = format!("{} {}", task.icon(), task.label());
            chip_btn(
                sheet,
                t,
                task_row,
                &label,
                self.task == Some(task),
                &ai_tag("task", task.label()),
            );
        }

        // Model (filtered by task)
        if let Some(task) = self.task {
            t.add_text(
                panel,
                "2. Model",
                s(sheet, "font-11").color(if self.stage == PlannerStage::Model {
                    theme::ACCENT
                } else {
                    theme::OVERLAY0
                }),
            );
            let model_row = t.add_box(panel, s(sheet, "wrap-row"));
            for &model in ModelKind::for_task(task) {
                chip_btn(
                    sheet,
                    t,
                    model_row,
                    model.label(),
                    self.model == Some(model),
                    &ai_tag("model", model.label()),
                );
            }
            if let Some(model) = self.model {
                let card = t.add_box(
                    panel,
                    Style::default()
                        .bg(theme::SURFACE0)
                        .radius(8.0)
                        .pad(10.0)
                        .gap(4.0),
                );
                t.add_text(card, model.label(), s(sheet, "heading"));
                t.add_text(card, model.family(), s(sheet, "label"));
                let row = t.add_box(card, Style::default().row().gap(6.0).align(Align::Center));
                t.add_text(row, task.label(), badge(sheet, "blue"));
                let view_btn = t.add_box(
                    row,
                    Style::default()
                        .h(22.0)
                        .radius(6.0)
                        .pad_xy(8.0, 0.0)
                        .bg(theme::SURFACE_BRIGHT)
                        .align(Align::Center)
                        .justify(Justify::Center)
                        .cursor(Cursor::Pointer),
                );
                t.add_text(
                    view_btn,
                    "View Architecture",
                    s(sheet, "font-9").color(theme::ACCENT),
                );
                t.tag(view_btn, &ai_tag("view", model.label()));
            }
        }

        // Dataset (filtered by task)
        if let Some(task) = self.task {
            if self.model.is_some() {
                t.add_text(
                    panel,
                    "3. Dataset",
                    s(sheet, "font-11").color(if self.stage == PlannerStage::Dataset {
                        theme::ACCENT
                    } else {
                        theme::OVERLAY0
                    }),
                );
                let ds_row = t.add_box(panel, s(sheet, "wrap-row"));
                for &ds in DatasetSource::for_task(task) {
                    chip_btn(
                        sheet,
                        t,
                        ds_row,
                        ds.label(),
                        self.dataset == Some(ds),
                        &ai_tag("ds", ds.label()),
                    );
                }
                if let Some(ds) = self.dataset {
                    t.add_text(panel, ds.size_info(), s(sheet, "detail"));
                }
            }
        }

        // Start button
        if self.model.is_some() && self.dataset.is_some() && !self.training {
            let start = t.add_box(
                panel,
                Style::default()
                    .h(32.0)
                    .radius(8.0)
                    .bg(theme::GREEN)
                    .align(Align::Center)
                    .justify(Justify::Center)
                    .cursor(Cursor::Pointer),
            );
            t.add_text(
                start,
                "Start Training",
                s(sheet, "font-11").color(theme::SIDEBAR_BG),
            );
            t.tag(start, "ai-start-train");
        }

        // Progress
        if self.training {
            let prog = t.add_box(
                panel,
                Style::default()
                    .bg(theme::SURFACE0)
                    .radius(6.0)
                    .pad(8.0)
                    .gap(4.0),
            );
            t.add_text(
                prog,
                &format!("Epoch {}/{}", self.epoch, self.total_epochs),
                s(sheet, "subheading"),
            );
            let pct = self.epoch as f64 / self.total_epochs as f64;
            let bar_track = t.add_box(prog, s(sheet, "progress-track"));
            t.add_box(
                bar_track,
                Style::default()
                    .h(10.0)
                    .w(pct * 200.0)
                    .radius(5.0)
                    .bg(theme::ACCENT),
            );
        }
    }

    /// Viewer panel (center — loss curve + metrics + logs).
    fn build_viewer(&mut self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(parent, Style::default().grow(1.0).gap(12.0).pad(8.0));

        // Clear graph prims — we'll rebuild the loss chart below if needed
        self.prims = RenderList::default();

        if self.loss_history.is_empty() && !self.training {
            let center = t.add_box(
                panel,
                Style::default()
                    .grow(1.0)
                    .align(Align::Center)
                    .justify(Justify::Center),
            );
            t.add_text(
                center,
                "Select a task, model, and dataset to begin",
                s(sheet, "font-13").color(theme::OVERLAY0),
            );
            return;
        }

        // Loss curve
        t.add_text(panel, "Loss", s(sheet, "subheading"));
        let chart_box = t.add_box(
            panel,
            Style::default().h(180.0).bg(theme::SURFACE0).radius(8.0),
        );
        t.tag(chart_box, "ai-loss-chart");
        if !self.loss_history.is_empty() {
            let color = Color::rgb(137, 180, 250);
            self.prims
                .push_line_chart(0.0, 0.0, 500.0, 170.0, &self.loss_history, color, 3.0);
        }

        // Metrics
        if let Some(&last_loss) = self.loss_history.last() {
            let metrics_row = t.add_box(panel, sm(sheet, &["row-gap-12"]));
            let m1 = t.add_box(metrics_row, s(sheet, "metric-card"));
            t.add_text(m1, "Current Loss", s(sheet, "label"));
            t.add_text(
                m1,
                &format!("{last_loss:.4}"),
                sm(sheet, &["font-14", "text"]),
            );

            let m2 = t.add_box(metrics_row, s(sheet, "metric-card"));
            t.add_text(m2, "Epochs", s(sheet, "label"));
            t.add_text(
                m2,
                &format!("{}/{}", self.epoch, self.total_epochs),
                sm(sheet, &["font-14", "text"]),
            );

            let min_loss = self
                .loss_history
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let m3 = t.add_box(metrics_row, s(sheet, "metric-card"));
            t.add_text(m3, "Best Loss", s(sheet, "label"));
            t.add_text(
                m3,
                &format!("{min_loss:.4}"),
                sm(sheet, &["font-14", "green"]),
            );
        }

        // Logs
        t.add_text(panel, "Logs", s(sheet, "subheading"));
        let log_box = t.add_box(
            panel,
            Style::default()
                .grow(1.0)
                .bg(theme::SURFACE0)
                .radius(6.0)
                .pad(8.0)
                .gap(1.0)
                .overflow(Overflow::Scroll),
        );
        for line in self.logs.iter().rev().take(50) {
            t.add_text(log_box, line, s(sheet, "detail"));
        }
    }

    /// Graph panel (center) — interactive model architecture graph.
    fn build_graph_panel(&mut self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(parent, Style::default().grow(1.0).gap(4.0));

        // Header: title + back button
        let hdr = t.add_box(
            panel,
            Style::default()
                .row()
                .gap(8.0)
                .align(Align::Center)
                .pad_xy(8.0, 4.0),
        );
        let back = t.add_box(
            hdr,
            Style::default()
                .h(24.0)
                .radius(6.0)
                .pad_xy(10.0, 0.0)
                .bg(theme::SURFACE0)
                .align(Align::Center)
                .justify(Justify::Center)
                .cursor(Cursor::Pointer),
        );
        t.add_text(
            back,
            "\u{2190} Training",
            s(sheet, "font-9").color(theme::SUBTEXT0),
        );
        t.tag(back, "ai-back-train");

        let label = self.model.map_or("Model Architecture", |m| m.label());
        t.add_text(hdr, label, s(sheet, "heading"));

        // Graph viewport
        let viewport = t.add_box(panel, s(sheet, "scene-panel").grow(1.0));
        t.tag(viewport, VIEWPORT_TAG);

        // Render the graph into prims
        self.render_graph();
    }

    /// Models panel — model catalog with cards.
    fn build_models(&self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(
            parent,
            Style::default()
                .grow(1.0)
                .pad(12.0)
                .gap(8.0)
                .overflow(Overflow::Scroll),
        );

        t.add_text(panel, "Models", s(sheet, "heading"));
        let grid = t.add_box(panel, s(sheet, "wrap-row"));
        for model in ModelKind::ALL {
            let is_selected = self.model == Some(model);
            let bg = if is_selected {
                theme::SURFACE_BRIGHT
            } else {
                theme::SURFACE0
            };
            let card = t.add_box(
                grid,
                Style::default()
                    .w(200.0)
                    .bg(bg)
                    .radius(8.0)
                    .pad(10.0)
                    .gap(4.0),
            );
            t.add_text(card, model.label(), sm(sheet, &["font-13", "text"]));
            let info_row = t.add_box(card, Style::default().row().gap(4.0).align(Align::Center));
            t.add_text(
                info_row,
                model.family(),
                s(sheet, "font-9").color(theme::OVERLAY0),
            );
            t.add_text(info_row, model.task().label(), badge(sheet, "blue"));
            // View Architecture button
            let btn = t.add_box(
                card,
                Style::default()
                    .h(24.0)
                    .radius(6.0)
                    .bg(if is_selected {
                        theme::ACCENT
                    } else {
                        theme::SURFACE_BRIGHT
                    })
                    .align(Align::Center)
                    .justify(Justify::Center)
                    .cursor(Cursor::Pointer),
            );
            let btn_fg = if is_selected {
                theme::SIDEBAR_BG
            } else {
                theme::ACCENT
            };
            t.add_text(btn, "View Architecture", s(sheet, "font-9").color(btn_fg));
            t.tag(btn, &ai_tag("view", model.label()));
        }

        // Runs section
        if !self.runs.is_empty() {
            t.add_text(panel, "Runs", s(sheet, "heading"));
            for (i, run) in self.runs.iter().enumerate().rev().take(10) {
                let bg = if i % 2 == 0 {
                    theme::SURFACE0
                } else {
                    Color::TRANSPARENT
                };
                let row = t.add_box(panel, Style::default().bg(bg).radius(4.0).pad(6.0).gap(2.0));
                t.add_text(
                    row,
                    &format!("{} — {}", run.model, run.dataset),
                    sm(sheet, &["font-11", "text"]),
                );
                t.add_text(
                    row,
                    &format!("{}ep · loss {:.4}", run.epochs, run.final_loss),
                    s(sheet, "font-9").color(theme::GREEN),
                );
            }
        }
    }

    /// Datasets panel — dataset catalog with detail views.
    fn build_datasets(&self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(
            parent,
            Style::default()
                .grow(1.0)
                .pad(12.0)
                .gap(8.0)
                .overflow(Overflow::Scroll),
        );

        if let Some(ds) = self.selected_dataset {
            self.build_ds_detail(sheet, t, panel, ds);
        } else {
            self.build_ds_catalog(sheet, t, panel);
        }
    }

    /// Catalog: clickable dataset cards with thumbnails.
    fn build_ds_catalog(&self, sheet: &StyleSheet, t: &mut Tree, panel: NodeId) {
        t.add_text(panel, "Datasets", s(sheet, "heading"));
        t.add_text(
            panel,
            "Click a dataset to explore its contents",
            s(sheet, "detail"),
        );
        let grid = t.add_box(panel, s(sheet, "wrap-row"));
        for ds in DatasetSource::ALL {
            let is_selected = self.dataset == Some(ds);
            let bg = if is_selected {
                theme::SURFACE_BRIGHT
            } else {
                theme::SURFACE0
            };
            let card = t.add_box(
                grid,
                Style::default()
                    .w(200.0)
                    .bg(bg)
                    .radius(8.0)
                    .pad(10.0)
                    .gap(6.0)
                    .cursor(Cursor::Pointer),
            );

            // Thumbnail preview area
            let thumb = t.add_box(
                card,
                Style::default()
                    .h(48.0)
                    .bg(theme::SURFACE_BRIGHT)
                    .radius(4.0)
                    .row()
                    .gap(2.0)
                    .pad(4.0)
                    .align(Align::Center),
            );
            self.build_thumbnail_row(sheet, t, thumb, ds, 4);

            t.add_text(card, ds.label(), sm(sheet, &["font-13", "text"]));
            t.add_text(
                card,
                ds.size_info(),
                s(sheet, "font-9").color(theme::OVERLAY0),
            );
            // Task compatibility badges
            let badge_row = t.add_box(card, Style::default().row().gap(4.0).align(Align::Center));
            for task in TaskType::ALL {
                if DatasetSource::for_task(task).contains(&ds) {
                    t.add_text(badge_row, task.label(), badge(sheet, "blue"));
                }
            }
            t.tag(card, &ai_tag("dataset", ds.label()));
        }
    }

    /// Dataset detail view — header, stats, view tabs, and content.
    fn build_ds_detail(&self, sheet: &StyleSheet, t: &mut Tree, panel: NodeId, ds: DatasetSource) {
        // ── Header: back button + name + version selector ──
        let hdr = t.add_box(panel, Style::default().row().gap(8.0).align(Align::Center));
        let back = t.add_box(
            hdr,
            Style::default()
                .h(24.0)
                .radius(6.0)
                .pad_xy(10.0, 0.0)
                .bg(theme::SURFACE0)
                .align(Align::Center)
                .justify(Justify::Center)
                .cursor(Cursor::Pointer),
        );
        t.add_text(
            back,
            "\u{2190} Datasets",
            s(sheet, "font-9").color(theme::SUBTEXT0),
        );
        t.tag(back, "ai-ds-back");
        t.add_text(hdr, ds.label(), s(sheet, "heading"));
        t.add_text(
            hdr,
            ds.size_info(),
            s(sheet, "font-9").color(theme::OVERLAY0),
        );

        // Version selector chips
        let versions = ds_versions(ds);
        let ver_bar = t.add_box(hdr, Style::default().row().gap(2.0).align(Align::Center));
        for (i, label) in versions.iter().enumerate() {
            chip_btn(
                sheet,
                t,
                ver_bar,
                label,
                self.ds_version == i as u8,
                &format!("ai-ds-version-{i}"),
            );
        }

        // ── Stats bar ──
        let stats_bar = t.add_box(panel, Style::default().row().gap(8.0).align(Align::Center));
        let info = ds_info(ds);
        for (label, value) in &info.stats {
            kv_card(t, stats_bar, sheet, label, value);
        }

        // ── View mode subtab bar ──
        let view_bar = t.add_box(panel, Style::default().row().gap(2.0));
        const DS_VIEWS: &[(&str, &str, DatasetView)] = &[
            ("Cards", "ai-dsview-cards", DatasetView::Cards),
            ("Table", "ai-dsview-table", DatasetView::Table),
            ("Samples", "ai-dsview-samples", DatasetView::Samples),
            (
                "Embeddings",
                "ai-dsview-embeddings",
                DatasetView::Embeddings,
            ),
            (
                "Transforms",
                "ai-dsview-transforms",
                DatasetView::Transforms,
            ),
        ];
        for &(label, tag, view) in DS_VIEWS {
            chip_btn(sheet, t, view_bar, label, self.dataset_view == view, tag);
        }

        // ── Filter bar (shown on Cards/Table/Samples) ──
        if matches!(
            self.dataset_view,
            DatasetView::Cards | DatasetView::Table | DatasetView::Samples
        ) {
            let filter_row = t.add_box(panel, Style::default().row().gap(4.0).align(Align::Center));
            t.add_text(
                filter_row,
                "Filter:",
                s(sheet, "font-9").color(theme::OVERLAY0),
            );
            chip_btn(
                sheet,
                t,
                filter_row,
                "All",
                self.ds_filter_class.is_none(),
                "ai-ds-filter-clear",
            );
            for (i, cls) in info.classes.iter().enumerate() {
                chip_btn(
                    sheet,
                    t,
                    filter_row,
                    cls,
                    self.ds_filter_class == Some(i),
                    &format!("ai-ds-filter-{i}"),
                );
            }
        }

        // For Cards/Table/Samples views, use a split layout: content + viewer
        let has_viewer = self.ds_selected_sample.is_some()
            && matches!(
                self.dataset_view,
                DatasetView::Cards | DatasetView::Table | DatasetView::Samples
            );

        if has_viewer {
            let split = t.add_box(panel, Style::default().row().gap(8.0).grow(1.0));
            // Left: scrollable content area (overflow hidden to prevent pushing viewer)
            let content = t.add_box(
                split,
                Style::default()
                    .grow(1.0)
                    .gap(6.0)
                    .min_w(0.0)
                    .overflow(Overflow::Hidden),
            );
            match self.dataset_view {
                DatasetView::Cards => self.build_ds_cards(sheet, t, content, ds, &info),
                DatasetView::Table => self.build_ds_table(sheet, t, content, ds, &info),
                DatasetView::Samples => self.build_ds_samples(sheet, t, content, ds, &info),
                _ => {}
            }
            // Right: sample viewer panel
            self.build_sample_viewer(sheet, t, split, ds, &info);
        } else {
            match self.dataset_view {
                DatasetView::Cards => self.build_ds_cards(sheet, t, panel, ds, &info),
                DatasetView::Table => self.build_ds_table(sheet, t, panel, ds, &info),
                DatasetView::Samples => self.build_ds_samples(sheet, t, panel, ds, &info),
                DatasetView::Embeddings => self.build_ds_embeddings(sheet, t, panel, ds, &info),
                DatasetView::Transforms => self.build_ds_transforms(sheet, t, panel, ds),
            }
        }
    }

    /// Sample viewer — enlarged pixel grid + full metadata sidebar.
    fn build_sample_viewer(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        info: &DatasetInfo,
    ) {
        let sel = match self.ds_selected_sample {
            Some(s) => s,
            None => return,
        };
        let samples = ds_samples(ds);
        let sample = match samples.get(sel) {
            Some(s) => s,
            None => return,
        };

        let viewer = t.add_box(
            parent,
            Style::default()
                .w(220.0)
                .min_w(220.0)
                .bg(theme::SURFACE0)
                .radius(8.0)
                .pad(12.0)
                .gap(8.0),
        );

        // Close button
        let close = t.add_box(
            viewer,
            Style::default()
                .row()
                .gap(4.0)
                .align(Align::Center)
                .cursor(Cursor::Pointer),
        );
        t.add_text(
            close,
            &format!("Sample #{sel}"),
            s(sheet, "font-13").color(theme::TEXT),
        );
        t.tag(close, "ai-ds-viewer-close");

        // Enlarged pixel grid
        let grid_n = ds_grid_size(ds);
        if grid_n > 0 {
            let img_sz = 180.0;
            let img_box = t.add_box(
                viewer,
                Style::default()
                    .wh(img_sz, img_sz)
                    .bg(Color::rgba(0, 0, 0, 255))
                    .radius(4.0)
                    .align_self(Align::Center),
            );
            self.build_pixel_grid(sheet, t, img_box, ds, sample, img_sz);
        }

        // Class info
        let cls_color = sample.class_color();
        let cls_label = sample.class_label(info);
        let cls_row = t.add_box(viewer, Style::default().row().gap(6.0).align(Align::Center));
        t.add_box(
            cls_row,
            Style::default().wh(10.0, 10.0).bg(cls_color).radius(5.0),
        );
        t.add_text(cls_row, cls_label, s(sheet, "font-11").color(cls_color));

        // Metadata rows
        let meta_items = [
            ("Name", sample.name.to_string()),
            ("Size", sample.size_label.clone()),
            ("Type", ds_sample_type(ds).to_string()),
            ("Index", format!("#{sel}")),
        ];
        for (label, value) in &meta_items {
            let row = t.add_box(viewer, Style::default().row().gap(4.0));
            t.add_text(row, *label, s(sheet, "font-9").color(theme::OVERLAY0));
            t.add_text(row, value.as_str(), s(sheet, "font-9").color(theme::TEXT));
        }

        // Preview text
        t.add_text(
            viewer,
            &sample.preview,
            s(sheet, "font-9").color(theme::SUBTEXT0),
        );
    }

    /// Build a row of small thumbnail previews.
    fn build_thumbnail_row(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        count: usize,
    ) {
        let samples = ds_samples(ds);
        for (i, sample) in samples.iter().take(count).enumerate() {
            let sz = 32.0;
            let pixel_box = t.add_box(
                parent,
                Style::default().wh(sz, sz).bg(theme::SURFACE0).radius(2.0),
            );
            self.build_pixel_grid(sheet, t, pixel_box, ds, sample, sz);
            // Overlay index
            t.add_text(
                pixel_box,
                &format!("#{i}"),
                s(sheet, "font-9").color(Color::rgba(255, 255, 255, 180)),
            );
        }
    }

    /// Build a pixel grid for a sample inside a box of given size.
    fn build_pixel_grid(
        &self,
        _sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        sample: &SampleRecord,
        sz: f64,
    ) {
        let grid_n = ds_grid_size(ds);
        if grid_n == 0 {
            return;
        }
        let cell = (sz - 4.0) / grid_n as f64; // minus padding
        let grid = t.add_box(
            parent,
            Style::default()
                .wh(sz, sz)
                .row()
                .wrap(FlexWrap::Wrap)
                .pad(2.0),
        );
        let pixels = self.pixel_data(ds, sample.class_idx);
        for &color in pixels.iter().take(grid_n * grid_n) {
            t.add_box(grid, Style::default().wh(cell, cell).bg(color));
        }
    }

    /// Cards view — thumbnail grid with metadata.
    fn build_ds_cards(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        info: &DatasetInfo,
    ) {
        let grid = t.add_box(parent, s(sheet, "wrap-row"));
        let samples = ds_samples(ds);
        for (i, sample) in samples.iter().enumerate() {
            if !self.sample_matches_filter(sample, info) {
                continue;
            }
            let selected = self.ds_selected_sample == Some(i);
            let bg = if selected {
                theme::SURFACE_BRIGHT
            } else {
                theme::SURFACE0
            };
            let card = t.add_box(
                grid,
                Style::default()
                    .w(150.0)
                    .bg(bg)
                    .radius(8.0)
                    .pad(8.0)
                    .gap(4.0)
                    .cursor(Cursor::Pointer),
            );
            t.tag(card, &format!("ai-ds-sample-{i}"));

            // Pixel grid thumbnail
            let thumb_sz = 64.0;
            let thumb = t.add_box(
                card,
                Style::default()
                    .wh(thumb_sz, thumb_sz)
                    .bg(theme::SURFACE_BRIGHT)
                    .radius(4.0)
                    .align_self(Align::Center),
            );
            self.build_pixel_grid(sheet, t, thumb, ds, sample, thumb_sz);

            t.add_text(card, sample.name, sm(sheet, &["font-11", "text"]));
            let cls_color = sample.class_color();
            let cls_label = sample.class_label(info);
            t.add_text(card, cls_label, s(sheet, "font-9").color(cls_color));
            t.add_text(
                card,
                &sample.size_label,
                s(sheet, "font-9").color(theme::OVERLAY0),
            );
        }
    }

    /// Table view — sortable columns + filter.
    fn build_ds_table(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        info: &DatasetInfo,
    ) {
        let table = t.add_box(
            parent,
            Style::default()
                .bg(theme::SURFACE0)
                .radius(8.0)
                .pad(8.0)
                .gap(0.0)
                .grow(1.0),
        );

        // Sortable header
        let hdr = t.add_box(
            table,
            Style::default()
                .row()
                .gap(4.0)
                .pad_xy(8.0, 4.0)
                .align(Align::Center),
        );
        let cols = ds_table_columns(ds);
        for (ci, &(label, w)) in cols.iter().enumerate() {
            let arrow = if self.ds_sort_col == ci {
                if self.ds_sort_asc {
                    " \u{25B2}"
                } else {
                    " \u{25BC}"
                }
            } else {
                ""
            };
            let col_label = format!("{label}{arrow}");
            let col_btn = t.add_box(
                hdr,
                Style::default()
                    .w(w)
                    .h(20.0)
                    .cursor(Cursor::Pointer)
                    .align(Align::Center),
            );
            t.add_text(
                col_btn,
                &col_label,
                s(sheet, "font-9").color(if self.ds_sort_col == ci {
                    theme::ACCENT
                } else {
                    theme::OVERLAY0
                }),
            );
            t.tag(col_btn, &format!("ai-ds-sort-{ci}"));
        }

        // Collect and sort samples
        let samples = ds_samples(ds);
        let mut indices: Vec<usize> = (0..samples.len())
            .filter(|&i| self.sample_matches_filter(&samples[i], info))
            .collect();
        indices.sort_by(|&a, &b| {
            let ord = match self.ds_sort_col {
                0 => a.cmp(&b),
                1 => samples[a].name.cmp(samples[b].name),
                2 => samples[a].class_idx.cmp(&samples[b].class_idx),
                _ => samples[a].size_label.cmp(&samples[b].size_label),
            };
            if self.ds_sort_asc { ord } else { ord.reverse() }
        });

        // Rows
        for (row_i, &i) in indices.iter().enumerate() {
            let sample = &samples[i];
            let selected = self.ds_selected_sample == Some(i);
            let bg = if selected {
                theme::ACCENT.with_alpha(30)
            } else if row_i % 2 == 0 {
                Color::TRANSPARENT
            } else {
                Color::rgba(255, 255, 255, 6)
            };
            let row = t.add_box(
                table,
                Style::default()
                    .row()
                    .gap(4.0)
                    .pad_xy(8.0, 2.0)
                    .bg(bg)
                    .align(Align::Center)
                    .cursor(Cursor::Pointer),
            );
            t.tag(row, &format!("ai-ds-sample-{i}"));

            t.add_text(
                row,
                &i.to_string(),
                s(sheet, "font-9").color(theme::SUBTEXT0).w(cols[0].1),
            );
            t.add_text(row, sample.name, s(sheet, "source-text").w(cols[1].1));
            let cls = sample.class_label(info);
            t.add_text(
                row,
                cls,
                s(sheet, "font-9").color(sample.class_color()).w(cols[2].1),
            );
            t.add_text(
                row,
                &sample.size_label,
                s(sheet, "font-9").color(theme::SUBTEXT0).w(cols[3].1),
            );
        }
    }

    /// Samples view — preview cards with pixel grids and metadata.
    fn build_ds_samples(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        info: &DatasetInfo,
    ) {
        let samples = ds_samples(ds);
        for (i, sample) in samples.iter().enumerate() {
            if !self.sample_matches_filter(sample, info) {
                continue;
            }
            let selected = self.ds_selected_sample == Some(i);
            let card_bg = if selected {
                theme::SURFACE_BRIGHT
            } else {
                theme::SURFACE0
            };
            let card = t.add_box(
                parent,
                Style::default()
                    .bg(card_bg)
                    .radius(8.0)
                    .pad(10.0)
                    .gap(6.0)
                    .row()
                    .cursor(Cursor::Pointer),
            );
            t.tag(card, &format!("ai-ds-sample-{i}"));

            // Left: pixel grid thumbnail
            let thumb_sz = 80.0;
            let thumb = t.add_box(
                card,
                Style::default()
                    .wh(thumb_sz, thumb_sz)
                    .min_w(thumb_sz)
                    .bg(theme::SURFACE0)
                    .radius(4.0),
            );
            self.build_pixel_grid(sheet, t, thumb, ds, sample, thumb_sz);

            // Right: metadata column
            let meta = t.add_box(card, Style::default().grow(1.0).gap(3.0));
            let hdr = t.add_box(meta, Style::default().row().gap(8.0).align(Align::Center));
            let cls_color = sample.class_color();
            t.add_text(
                hdr,
                &format!("Sample #{i}"),
                s(sheet, "font-11").color(cls_color),
            );
            t.add_text(hdr, sample.name, sm(sheet, &["font-11", "text"]));
            t.add_text(hdr, &sample.size_label, badge(sheet, "green"));

            t.add_text(
                meta,
                &format!("Class: {}", sample.class_label(info)),
                s(sheet, "detail"),
            );
            t.add_text(
                meta,
                &format!("Type: {}", ds_sample_type(ds)),
                s(sheet, "detail"),
            );
            t.add_text(meta, &sample.preview, s(sheet, "font-9").color(theme::TEXT));
        }
    }

    /// Embeddings view — 2D scatter plot with class-colored dots and controls.
    fn build_ds_embeddings(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
        info: &DatasetInfo,
    ) {
        // Controls bar
        let controls = t.add_box(
            parent,
            Style::default()
                .row()
                .gap(8.0)
                .align(Align::Center)
                .pad_xy(0.0, 2.0),
        );
        t.add_text(
            controls,
            "Embedding Space",
            s(sheet, "font-13").color(theme::TEXT),
        );

        // Method selector chips
        let methods = ["t-SNE", "UMAP", "PCA"];
        for (i, &method) in methods.iter().enumerate() {
            chip_btn(
                sheet,
                t,
                controls,
                method,
                self.ds_embed_method == i as u8,
                &format!("ai-ds-embed-method-{i}"),
            );
        }

        // Perplexity control
        let perp_box = t.add_box(
            controls,
            Style::default().row().gap(4.0).align(Align::Center),
        );
        t.add_text(
            perp_box,
            "Perplexity:",
            s(sheet, "font-9").color(theme::OVERLAY0),
        );
        for &p in &[5u8, 15, 30, 50] {
            chip_btn(
                sheet,
                t,
                perp_box,
                &p.to_string(),
                self.ds_embed_perplexity == p,
                &format!("ai-ds-embed-perp-{p}"),
            );
        }

        // Split: scatter + legend
        let body = t.add_box(parent, Style::default().row().gap(8.0).grow(1.0));

        // Scatter viewport — prims composited by main.rs
        let viewport = t.add_box(
            body,
            Style::default()
                .grow(1.0)
                .min_h(300.0)
                .bg(theme::SURFACE0)
                .radius(8.0),
        );
        t.tag(viewport, "ai-ds-scatter");

        // Legend + selected sample panel
        let side = t.add_box(body, Style::default().w(160.0).min_w(160.0).gap(8.0));

        // Class legend
        let legend = t.add_box(
            side,
            Style::default()
                .bg(theme::SURFACE0)
                .radius(6.0)
                .pad(8.0)
                .gap(4.0),
        );
        t.add_text(legend, "Classes", s(sheet, "font-11").color(theme::TEXT));
        for (i, cls) in info.classes.iter().enumerate() {
            let dot = t.add_box(
                legend,
                Style::default()
                    .row()
                    .gap(6.0)
                    .align(Align::Center)
                    .cursor(Cursor::Pointer),
            );
            let filtered = self.ds_filter_class == Some(i);
            t.add_box(
                dot,
                Style::default()
                    .wh(10.0, 10.0)
                    .bg(class_color(i))
                    .radius(5.0),
            );
            let label_color = if filtered {
                theme::ACCENT
            } else {
                theme::SUBTEXT0
            };
            t.add_text(dot, *cls, s(sheet, "font-9").color(label_color));
            t.tag(dot, &format!("ai-ds-filter-{i}"));
        }

        // Selected sample detail
        if let Some(sel) = self.ds_selected_sample {
            let samples = ds_samples(ds);
            if let Some(sample) = samples.get(sel) {
                let detail = t.add_box(
                    side,
                    Style::default()
                        .bg(theme::SURFACE0)
                        .radius(6.0)
                        .pad(8.0)
                        .gap(6.0),
                );
                let cls_color = sample.class_color();
                t.add_text(
                    detail,
                    &format!("Sample #{sel}"),
                    s(sheet, "font-11").color(cls_color),
                );

                // Enlarged thumbnail in side panel
                let grid_n = ds_grid_size(ds);
                if grid_n > 0 {
                    let thumb_sz = 140.0;
                    let thumb = t.add_box(
                        detail,
                        Style::default()
                            .wh(thumb_sz, thumb_sz)
                            .bg(Color::rgba(0, 0, 0, 255))
                            .radius(4.0)
                            .align_self(Align::Center),
                    );
                    self.build_pixel_grid(sheet, t, thumb, ds, sample, thumb_sz);
                }

                t.add_text(detail, sample.name, sm(sheet, &["font-11", "text"]));
                let cls_label = sample.class_label(info);
                t.add_text(detail, cls_label, s(sheet, "font-9").color(cls_color));
                t.add_text(
                    detail,
                    &sample.size_label,
                    s(sheet, "font-9").color(theme::OVERLAY0),
                );
            }
        }
    }

    /// Transforms view — pipeline builder with toggleable augmentation chips.
    fn build_ds_transforms(
        &self,
        sheet: &StyleSheet,
        t: &mut Tree,
        parent: NodeId,
        ds: DatasetSource,
    ) {
        t.add_text(
            parent,
            "Transform Pipeline",
            s(sheet, "font-13").color(theme::TEXT),
        );
        t.add_text(
            parent,
            "Toggle augmentations to build a data transform pipeline",
            s(sheet, "detail"),
        );

        // Chips grid
        let grid = t.add_box(parent, s(sheet, "wrap-row"));
        for tfm in Transform::ALL {
            let active = self.ds_transforms & (1u8 << (tfm as u8)) != 0;
            chip_btn(
                sheet,
                t,
                grid,
                tfm.label(),
                active,
                &format!("ai-ds-tfm-{}", tfm.tag()),
            );
        }

        // Active pipeline summary
        let active_tfms: Vec<&str> = Transform::ALL
            .iter()
            .filter(|tfm| self.ds_transforms & (1u8 << (**tfm as u8)) != 0)
            .map(|tfm| tfm.label())
            .collect();

        let pipeline = t.add_box(
            parent,
            Style::default()
                .bg(theme::SURFACE0)
                .radius(8.0)
                .pad(12.0)
                .gap(6.0),
        );
        t.add_text(
            pipeline,
            "Active Pipeline",
            s(sheet, "font-11").color(theme::ACCENT),
        );
        if active_tfms.is_empty() {
            t.add_text(
                pipeline,
                "No transforms selected. Click chips above to add.",
                s(sheet, "detail"),
            );
        } else {
            let flow = t.add_box(
                pipeline,
                Style::default().row().gap(6.0).align(Align::Center),
            );
            for (i, &tfm_label) in active_tfms.iter().enumerate() {
                if i > 0 {
                    t.add_text(flow, "\u{2192}", s(sheet, "font-11").color(theme::OVERLAY0));
                }
                let step = t.add_box(
                    flow,
                    Style::default()
                        .h(24.0)
                        .pad_xy(8.0, 0.0)
                        .radius(4.0)
                        .bg(theme::ACCENT.with_alpha(30))
                        .align(Align::Center)
                        .justify(Justify::Center),
                );
                t.add_text(step, tfm_label, s(sheet, "font-9").color(theme::ACCENT));
            }
        }

        // Preview: show effect description
        let preview = t.add_box(
            parent,
            Style::default()
                .bg(theme::SURFACE0)
                .radius(8.0)
                .pad(12.0)
                .gap(4.0),
        );
        t.add_text(preview, "Preview", s(sheet, "font-11").color(theme::TEXT));
        let info = ds_info(ds);
        let total = info
            .stats
            .iter()
            .find(|(l, _)| *l == "Samples")
            .map(|(_, v)| v.as_str())
            .unwrap_or("N/A");
        let aug_factor = 1 + active_tfms.len();
        t.add_text(
            preview,
            &format!("Original: {total} samples \u{2192} Augmented: ~{total}\u{00D7}{aug_factor}"),
            s(sheet, "detail"),
        );
        t.add_text(
            preview,
            &format!("Effective dataset size: ~{}x original", aug_factor),
            s(sheet, "font-9").color(theme::GREEN),
        );
    }

    /// Check if a sample passes the current class filter.
    fn sample_matches_filter(&self, sample: &SampleRecord, info: &DatasetInfo) -> bool {
        match self.ds_filter_class {
            None => true,
            Some(cls) => sample.class_idx % info.classes.len() == cls,
        }
    }

    /// Build scatter plot prims for embedding view.
    pub fn build_scatter_prims(&mut self, vp_w: f64, vp_h: f64) {
        self.scatter_prims.clear();
        let ds = match self.selected_dataset {
            Some(ds) => ds,
            None => return,
        };
        if self.dataset_view != DatasetView::Embeddings {
            return;
        }

        let pad = 20.0;
        let plot_w = vp_w - pad * 2.0;
        let plot_h = vp_h - pad * 2.0;
        if plot_w <= 0.0 || plot_h <= 0.0 {
            return;
        }

        // Grid lines
        self.scatter_prims.push_grid(
            pad,
            pad,
            plot_w,
            plot_h,
            8,
            8,
            Color::rgba(255, 255, 255, 15),
            0.5,
        );

        // Axes
        self.scatter_prims.push_axes(
            pad,
            pad,
            plot_w,
            plot_h,
            Color::rgba(255, 255, 255, 40),
            1.0,
        );

        let samples = ds_samples(ds);
        let embeddings = ds_embeddings(ds, self.ds_embed_perplexity);
        let info = ds_info(ds);
        let grid_n = ds_grid_size(ds);

        for (i, sample) in samples.iter().enumerate() {
            if !self.sample_matches_filter(sample, &info) {
                continue;
            }

            let (ex, ey) = embeddings[i % embeddings.len()];
            let x = pad + ex * plot_w;
            let y = pad + ey * plot_h;
            let color = sample.class_color();
            let selected = self.ds_selected_sample == Some(i);

            let pixels = self.pixel_data(ds, sample.class_idx);
            let has_image = grid_n > 0 && !pixels.is_empty();

            if has_image {
                // Render a small pixel thumbnail centered on the point
                let thumb = if selected { 18.0 } else { 10.0 };
                let cell = thumb / grid_n as f64;
                let ox = x - thumb / 2.0;
                let oy = y - thumb / 2.0;

                if selected {
                    // Highlight border
                    self.scatter_prims.push_rect(
                        ox - 2.0,
                        oy - 2.0,
                        thumb + 4.0,
                        thumb + 4.0,
                        Color::rgba(255, 255, 255, 160),
                    );
                }
                for (px, &c) in pixels.iter().take(grid_n * grid_n).enumerate() {
                    let cr = px / grid_n;
                    let cc = px % grid_n;
                    self.scatter_prims.push_rect(
                        ox + cc as f64 * cell,
                        oy + cr as f64 * cell,
                        cell.ceil(),
                        cell.ceil(),
                        c,
                    );
                }
            } else {
                // Fallback: colored dot
                let radius = if selected { 6.0 } else { 4.0 };
                if selected {
                    self.scatter_prims.push_circle(
                        x,
                        y,
                        radius + 2.0,
                        Color::rgba(255, 255, 255, 100),
                    );
                }
                self.scatter_prims.push_circle(x, y, radius, color);
            }
        }
    }
}


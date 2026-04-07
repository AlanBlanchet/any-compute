use super::*;
/// What the center panel is displaying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenterView {
    /// Training dashboard (loss chart, metrics, logs).
    Training,
    /// Model architecture graph (interactive zoom/pan/breadcrumb).
    ModelGraph,
    /// Model catalog.
    Models,
    /// Dataset catalog.
    Datasets,
}

/// How to display dataset samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetView {
    /// Card grid — thumbnail previews with metadata.
    Cards,
    /// Tabular listing with sortable columns.
    Table,
    /// Sample detail with pixel preview.
    Samples,
    /// 2D embedding scatter plot.
    Embeddings,
    /// Transform pipeline builder.
    Transforms,
}

/// Available data augmentation transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    Normalize,
    FlipH,
    FlipV,
    Rotate90,
    RandomCrop,
    GaussianNoise,
    ColorJitter,
    CutOut,
}

impl Transform {
    pub const ALL: [Self; 8] = [
        Self::Normalize,
        Self::FlipH,
        Self::FlipV,
        Self::Rotate90,
        Self::RandomCrop,
        Self::GaussianNoise,
        Self::ColorJitter,
        Self::CutOut,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Normalize => "Normalize",
            Self::FlipH => "Flip H",
            Self::FlipV => "Flip V",
            Self::Rotate90 => "Rotate 90\u{00B0}",
            Self::RandomCrop => "Random Crop",
            Self::GaussianNoise => "Gaussian Noise",
            Self::ColorJitter => "Color Jitter",
            Self::CutOut => "CutOut",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Self::Normalize => "normalize",
            Self::FlipH => "flip-h",
            Self::FlipV => "flip-v",
            Self::Rotate90 => "rotate-90",
            Self::RandomCrop => "random-crop",
            Self::GaussianNoise => "gaussian-noise",
            Self::ColorJitter => "color-jitter",
            Self::CutOut => "cut-out",
        }
    }

    pub fn from_tag(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.tag() == s)
    }
}

/// Center view tab labels + tags for the tab bar.
pub(super) const CENTER_TABS: &[(&str, &str)] = &[
    ("Training", "ai-center-training"),
    ("Models", "ai-center-models"),
    ("Datasets", "ai-center-datasets"),
];

/// Planner stage in the cascading workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerStage {
    Task,
    Model,
    Dataset,
    Training,
}

/// Saved training run.
pub struct RunEntry {
    pub model: &'static str,
    pub dataset: &'static str,
    pub epochs: usize,
    pub final_loss: f64,
}

/// Unified AI tab state — bundles training planner, simulation, and graph
/// interaction into one struct (same pattern as `SceneInfo` for 3D tab).
pub struct AiState {
    // ── planner ──
    pub stage: PlannerStage,
    pub task: Option<TaskType>,
    pub model: Option<ModelKind>,
    pub dataset: Option<DatasetSource>,

    // ── center view ──
    pub center: CenterView,

    // ── dataset viewer ──
    pub selected_dataset: Option<DatasetSource>,
    pub dataset_view: DatasetView,
    pub ds_sort_col: usize,
    pub ds_sort_asc: bool,
    pub ds_filter_class: Option<usize>,
    pub ds_version: u8,
    pub ds_transforms: u8,
    pub ds_selected_sample: Option<usize>,
    pub ds_embed_method: u8,
    pub ds_embed_perplexity: u8,
    pub scatter_prims: RenderList,

    // ── dataset provider (live fetch from HuggingFace etc.) ──
    pub(super) ds_provider_pixels: HashMap<DatasetSource, Vec<Vec<Color>>>,
    pub(super) ds_fetch_rx: Option<(
        DatasetSource,
        Receiver<Result<(DatasetMeta, VecSource), String>>,
    )>,
    pub(super) ds_load_error: Option<String>,

    // ── training sim ──
    pub training: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss_history: Vec<f64>,
    pub logs: Vec<String>,
    pub(super) epoch_timer: f64,
    pub runs: Vec<RunEntry>,

    // ── graph interaction (zoom / pan / breadcrumb) ──
    pub view: GraphView,
    pub prims: RenderList,
    pub drag_active: bool,
    pub drag_last: (f64, f64),
    pub click_pos: Option<(f64, f64)>,
    pub(super) cached_visual: Option<CachedGraphInfo>,
}

/// Cached graph visual — computed once per source, reused every frame.
pub struct CachedGraphInfo {
    pub visual: VisualGraph<2>,
    pub source: usize,
}

impl Default for AiState {
    fn default() -> Self {
        Self {
            stage: PlannerStage::Task,
            task: None,
            model: None,
            dataset: None,
            center: CenterView::Training,
            selected_dataset: None,
            dataset_view: DatasetView::Cards,
            ds_sort_col: 0,
            ds_sort_asc: true,
            ds_filter_class: None,
            ds_version: 0,
            ds_transforms: 0,
            ds_selected_sample: None,
            ds_embed_method: 0,
            ds_embed_perplexity: 30,
            scatter_prims: RenderList::default(),
            ds_provider_pixels: HashMap::new(),
            ds_fetch_rx: None,
            ds_load_error: None,
            training: false,
            epoch: 0,
            total_epochs: 10,
            loss_history: Vec::new(),
            logs: Vec::new(),
            epoch_timer: 0.0,
            runs: Vec::new(),
            view: GraphView::default(),
            prims: RenderList::default(),
            drag_active: false,
            drag_last: (0.0, 0.0),
            click_pos: None,
            cached_visual: None,
        }
    }
}

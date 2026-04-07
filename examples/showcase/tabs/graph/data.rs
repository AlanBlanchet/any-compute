use super::*;

/// A single sample record with all metadata.
pub struct SampleRecord {
    pub name: &'static str,
    pub class_idx: usize,
    pub size_label: String,
    pub preview: String,
}

impl SampleRecord {
    /// Class label from dataset info, safe modular index.
    pub(super) fn class_label<'a>(&self, info: &'a DatasetInfo) -> &'a str {
        info.classes[self.class_idx % info.classes.len()]
    }

    /// Class color.
    pub(super) fn class_color(&self) -> Color {
        class_color(self.class_idx)
    }
}

/// Aggregate dataset info.
pub struct DatasetInfo {
    pub classes: Vec<&'static str>,
    pub stats: Vec<(&'static str, String)>,
}

pub(super) fn ds_info(ds: DatasetSource) -> DatasetInfo {
    match ds {
        DatasetSource::Cifar10 => DatasetInfo {
            classes: vec![
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            stats: vec![
                ("Samples", "60,000".into()),
                ("Classes", "10".into()),
                ("Dimensions", "32\u{00D7}32\u{00D7}3".into()),
                ("Split", "50K/10K".into()),
            ],
        },
        DatasetSource::Mnist => DatasetInfo {
            classes: vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            stats: vec![
                ("Samples", "70,000".into()),
                ("Classes", "10".into()),
                ("Dimensions", "28\u{00D7}28".into()),
                ("Split", "60K/10K".into()),
            ],
        },
        DatasetSource::TinyShakespeare => DatasetInfo {
            classes: vec!["dialogue", "monologue", "sonnet", "stage-dir"],
            stats: vec![
                ("Samples", "40,000".into()),
                ("Vocab", "65 chars".into()),
                ("Total", "1.1M chars".into()),
                ("Lines", "~40K".into()),
            ],
        },
        DatasetSource::Synthetic => DatasetInfo {
            classes: vec!["class-A", "class-B", "class-C"],
            stats: vec![
                ("Samples", "\u{221E}".into()),
                ("Dim", "2D".into()),
                ("Noise", "0.05-0.15".into()),
                ("Patterns", "6".into()),
            ],
        },
    }
}

pub(super) fn ds_versions(ds: DatasetSource) -> &'static [&'static str] {
    match ds {
        DatasetSource::Cifar10 => &["v3 (latest)", "v2", "v1"],
        DatasetSource::Mnist => &["v2 (latest)", "v1"],
        DatasetSource::TinyShakespeare => &["v1"],
        DatasetSource::Synthetic => &["v2 (latest)", "v1"],
    }
}

pub(super) fn ds_sample_type(ds: DatasetSource) -> &'static str {
    match ds {
        DatasetSource::Cifar10 => "32\u{00D7}32 RGB",
        DatasetSource::Mnist => "28\u{00D7}28 Gray",
        DatasetSource::TinyShakespeare => "Text",
        DatasetSource::Synthetic => "2D Points",
    }
}

pub(super) fn ds_table_columns(ds: DatasetSource) -> Vec<(&'static str, f64)> {
    match ds {
        DatasetSource::Cifar10 | DatasetSource::Mnist => vec![
            ("#", 40.0),
            ("Name", 160.0),
            ("Class", 100.0),
            ("Size", 80.0),
        ],
        DatasetSource::TinyShakespeare => vec![
            ("#", 40.0),
            ("Section", 160.0),
            ("Type", 100.0),
            ("Length", 80.0),
        ],
        DatasetSource::Synthetic => vec![
            ("#", 40.0),
            ("Pattern", 160.0),
            ("Class", 100.0),
            ("Points", 80.0),
        ],
    }
}

/// Grid size for pixel thumbnails (0 = no grid, text-only datasets).
pub(super) fn ds_grid_size(ds: DatasetSource) -> usize {
    match ds {
        DatasetSource::Cifar10 => 12, // 12×12 coarse grid for 32×32
        DatasetSource::Mnist => 14,   // 14×14 coarse grid for 28×28
        DatasetSource::TinyShakespeare | DatasetSource::Synthetic => 0,
    }
}

/// Build all sample records for a dataset.
pub(super) fn ds_samples(ds: DatasetSource) -> Vec<SampleRecord> {
    let names = ds_sample_names(ds);
    let info = ds_info(ds);
    names
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let class_idx = i % info.classes.len();
            SampleRecord {
                name,
                class_idx,
                size_label: ds_sample_size(ds, i),
                preview: ds_sample_preview(ds, i),
            }
        })
        .collect()
}

pub(super) fn ds_sample_names(ds: DatasetSource) -> &'static [&'static str] {
    match ds {
        DatasetSource::Cifar10 => &[
            "airplane_001",
            "automobile_042",
            "bird_107",
            "cat_003",
            "deer_055",
            "dog_012",
            "frog_089",
            "horse_031",
            "ship_077",
            "truck_023",
        ],
        DatasetSource::Mnist => &[
            "digit_0_a",
            "digit_1_b",
            "digit_2_c",
            "digit_3_d",
            "digit_4_e",
            "digit_5_f",
            "digit_6_g",
            "digit_7_h",
            "digit_8_i",
            "digit_9_j",
        ],
        DatasetSource::TinyShakespeare => &[
            "act1_scene1",
            "act1_scene2",
            "act2_scene1",
            "act2_scene2",
            "act3_scene1",
            "sonnet_18",
            "sonnet_116",
            "soliloquy_hamlet",
        ],
        DatasetSource::Synthetic => &[
            "gaussian_2d",
            "spiral_3class",
            "moons_binary",
            "circles_nested",
            "linear_sep",
            "xor_pattern",
        ],
    }
}

pub(super) fn ds_sample_size(ds: DatasetSource, idx: usize) -> String {
    match ds {
        DatasetSource::Cifar10 => format!("{} KB", 3 + idx % 2),
        DatasetSource::Mnist => format!("{} B", 784 + idx * 10),
        DatasetSource::TinyShakespeare => format!("{} chars", 500 + idx * 137),
        DatasetSource::Synthetic => format!("{} pts", 100 + idx * 50),
    }
}

pub(super) fn ds_sample_preview(ds: DatasetSource, idx: usize) -> String {
    match ds {
        DatasetSource::Cifar10 => {
            let classes = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ];
            format!(
                "[32\u{00D7}32 RGB] class={} confidence=0.{:02}",
                classes[idx % classes.len()],
                85 + idx % 15
            )
        }
        DatasetSource::Mnist => format!(
            "[28\u{00D7}28 grayscale] label={} pixels=784 mean=0.{:02}",
            idx % 10,
            13 + idx * 3
        ),
        DatasetSource::TinyShakespeare => {
            let lines = [
                "To be, or not to be, that is the question",
                "All the world's a stage, and all the men and women merely players",
                "Now is the winter of our discontent",
                "Friends, Romans, countrymen, lend me your ears",
                "But soft, what light through yonder window breaks?",
                "Shall I compare thee to a summer's day?",
                "Let me not to the marriage of true minds admit impediments",
                "Whether 'tis nobler in the mind to suffer",
            ];
            lines[idx % lines.len()].to_string()
        }
        DatasetSource::Synthetic => format!(
            "[{} points] dim=2 classes={} noise=0.{:02}",
            100 + idx * 50,
            2 + idx % 3,
            5 + idx * 2
        ),
    }
}

/// Deterministic 2D embedding coordinates (fake t-SNE) for each sample.
/// Uses embed_perplexity to vary cluster separation.
pub(super) fn ds_embeddings(ds: DatasetSource, perplexity: u8) -> Vec<(f64, f64)> {
    let samples = ds_sample_names(ds);
    let n = samples.len();
    let info = ds_info(ds);
    let nc = info.classes.len();
    // Higher perplexity → tighter clusters; lower → more spread
    let spread = 0.04 + 0.16 * (1.0 - (perplexity as f64 / 50.0).min(1.0));
    (0..n)
        .map(|i| {
            let cls = i % nc;
            // Arrange clusters in a circle for better separation
            let angle = std::f64::consts::TAU * cls as f64 / nc as f64;
            let radius = 0.3;
            let cx = 0.5 + radius * angle.cos();
            let cy = 0.5 + radius * angle.sin();
            // Deterministic jitter within cluster
            let jx = ((i * 137 + cls * 53) % 100) as f64 / 100.0 * spread - spread / 2.0;
            let jy = ((i * 89 + cls * 41) % 100) as f64 / 100.0 * spread - spread / 2.0;
            ((cx + jx).clamp(0.02, 0.98), (cy + jy).clamp(0.02, 0.98))
        })
        .collect()
}

// ── Provider pixel decoding ────────────────────────────────────────────
// Convert raw pixel bytes from a DatasetProvider response into per-class
// Color grids suitable for the thumbnail renderer.

pub(super) fn decode_provider_pixels(
    ds: DatasetSource,
    meta: &DatasetMeta,
    source: &VecSource,
) -> Vec<Vec<Color>> {
    let shape = meta.image_shape.unwrap_or(ImageShape {
        width: 28,
        height: 28,
        channels: 1,
    });
    let src_w = shape.width as usize;
    let src_h = shape.height as usize;
    let ch = shape.channels as usize;
    let grid_n = ds_grid_size(ds).max(1); // match display grid exactly

    let num_classes = meta.classes.len().max(1);
    let mut per_class: Vec<Vec<Color>> = vec![vec![]; num_classes];

    for row in &source.rows {
        let label_idx = row.get(1).and_then(|c| c.as_i64()).unwrap_or(0) as usize;
        if label_idx >= num_classes {
            continue;
        }
        // Only keep first sample per class
        if !per_class[label_idx].is_empty() {
            continue;
        }
        let raw = match row.first() {
            Some(any_compute_core::data::CellValue::Bytes(b)) => b.as_slice(),
            _ => continue,
        };
        if raw.len() < src_w * src_h * ch {
            continue;
        }
        // Nearest-neighbor scale to grid_n × grid_n
        let colors: Vec<Color> = (0..grid_n * grid_n)
            .map(|px| {
                let r = px / grid_n * src_h / grid_n;
                let c = px % grid_n * src_w / grid_n;
                match ch {
                    1 => {
                        let v = raw[r * src_w + c];
                        Color::rgba(v, v, v, 255)
                    }
                    _ => {
                        let idx = (r * src_w + c) * ch;
                        Color::rgba(
                            raw[idx],
                            raw.get(idx + 1).copied().unwrap_or(0),
                            raw.get(idx + 2).copied().unwrap_or(0),
                            255,
                        )
                    }
                }
            })
            .collect();
        per_class[label_idx] = colors;
    }
    per_class
}

/// Per-class color palette for scatter dots and class labels.
pub(super) fn class_color(class_idx: usize) -> Color {
    const PALETTE: [Color; 10] = [
        theme::BLUE,
        theme::GREEN,
        theme::MAUVE,
        theme::YELLOW,
        theme::PEACH,
        theme::RED,
        theme::TEAL,
        theme::PINK,
        theme::FLAMINGO,
        theme::LAVENDER,
    ];
    PALETTE[class_idx % PALETTE.len()]
}

// ── Helpers ─────────────────────────────────────────────────────────────

pub(super) fn chip_btn(
    sheet: &StyleSheet,
    t: &mut Tree,
    parent: NodeId,
    label: &str,
    active: bool,
    tag: &str,
) {
    let (bg, fg) = if active {
        (theme::ACCENT, theme::SIDEBAR_BG)
    } else {
        (theme::SURFACE0, theme::SUBTEXT0)
    };
    let btn = t.add_box(
        parent,
        Style::default()
            .h(28.0)
            .radius(14.0)
            .pad_xy(14.0, 0.0)
            .bg(bg)
            .align(Align::Center)
            .justify(Justify::Center)
            .cursor(Cursor::Pointer),
    );
    t.add_text(btn, label, s(sheet, "font-11").color(fg));
    t.tag(btn, tag);
}

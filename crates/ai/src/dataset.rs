//! Dataset providers — `DatasetProvider` trait, `HuggingFaceProvider`, image decoding.

use any_compute_core::data::VecSource;

// Re-export core data types used by providers.
pub use any_compute_core::data::{CellValue, ColumnKind, ColumnMeta};

/// Image dimensions for pixel-data datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageShape {
    pub width: u32,
    pub height: u32,
    pub channels: u8, // 1=grayscale, 3=RGB
}

/// Metadata returned by a provider alongside its data.
#[derive(Debug, Clone)]
pub struct DatasetMeta {
    pub name: String,
    pub classes: Vec<String>,
    pub total_samples: usize,
    pub image_shape: Option<ImageShape>,
}

/// A provider that can fetch dataset samples from an external source.
///
/// Providers parse remote/local data into the common `VecSource` format.
/// Image pixels land in `CellValue::Bytes`, labels in `Text`/`Int`.
pub trait DatasetProvider: Send + Sync {
    /// Unique provider identifier (e.g. "huggingface", "torchvision").
    fn id(&self) -> &str;

    /// Human-readable name.
    fn label(&self) -> &str;

    /// Fetch up to `limit` samples for the named dataset.
    /// Returns metadata + a VecSource with columns ["pixels", "label", ...].
    fn fetch_samples(
        &self,
        dataset: &str,
        limit: usize,
    ) -> Result<(DatasetMeta, VecSource), String>;

    /// List dataset IDs this provider can serve.
    fn available_datasets(&self) -> &[&str];
}

// ── HuggingFace provider ────────────────────────────────────────────────

#[cfg(feature = "dataset")]
mod hf_provider {
    use super::*;

    /// Provider that fetches dataset rows from the HuggingFace Hub API.
    pub struct HuggingFaceProvider {
        datasets: Vec<HfDatasetDef>,
    }

    struct HfDatasetDef {
        id: &'static str,
        repo: &'static str,
        config: &'static str,
        split: &'static str,
        label_col: &'static str,
        image_col: &'static str,
        image_shape: ImageShape,
        classes: &'static [&'static str],
    }

    impl HuggingFaceProvider {
        pub fn new() -> Self {
            Self {
                datasets: vec![
                    HfDatasetDef {
                        id: "mnist",
                        repo: "ylecun/mnist",
                        config: "mnist",
                        split: "test",
                        label_col: "label",
                        image_col: "image",
                        image_shape: ImageShape {
                            width: 28,
                            height: 28,
                            channels: 1,
                        },
                        classes: &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    },
                    HfDatasetDef {
                        id: "cifar10",
                        repo: "uoft-cs/cifar10",
                        config: "plain_text",
                        split: "test",
                        label_col: "label",
                        image_col: "img",
                        image_shape: ImageShape {
                            width: 32,
                            height: 32,
                            channels: 3,
                        },
                        classes: &[
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
                    },
                ],
            }
        }

        fn find_def(&self, dataset: &str) -> Option<&HfDatasetDef> {
            self.datasets.iter().find(|d| d.id == dataset)
        }
    }

    impl DatasetProvider for HuggingFaceProvider {
        fn id(&self) -> &str {
            "huggingface"
        }

        fn label(&self) -> &str {
            "Hugging Face Hub"
        }

        fn available_datasets(&self) -> &[&str] {
            &["mnist", "cifar10"]
        }

        fn fetch_samples(
            &self,
            dataset: &str,
            limit: usize,
        ) -> Result<(DatasetMeta, VecSource), String> {
            let def = self
                .find_def(dataset)
                .ok_or_else(|| format!("unknown dataset: {dataset}"))?;

            let url = format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset=0&length={}",
                def.repo, def.config, def.split, limit
            );
            log::info!("HuggingFace: fetching {url}");

            let body: String = ureq::get(&url)
                .call()
                .map_err(|e| format!("HTTP error: {e}"))?
                .body_mut()
                .read_to_string()
                .map_err(|e| format!("read error: {e}"))?;

            let json: serde_json::Value =
                serde_json::from_str(&body).map_err(|e| format!("JSON parse error: {e}"))?;

            let rows = json["rows"].as_array().ok_or("missing 'rows' array")?;

            let mut data_rows = Vec::with_capacity(rows.len());
            for row_wrapper in rows {
                let row = &row_wrapper["row"];

                let label = row[def.label_col].clone();
                let label_val = match &label {
                    serde_json::Value::Number(n) => CellValue::Int(n.as_i64().unwrap_or(0)),
                    serde_json::Value::String(s) => CellValue::Text(s.clone()),
                    _ => CellValue::Int(0),
                };

                let pixel_val = match &row[def.image_col] {
                    serde_json::Value::Object(img) => {
                        if let Some(serde_json::Value::String(src)) = img.get("src") {
                            if let Some(b64) = src.split(",").nth(1) {
                                use base64::Engine;
                                base64::engine::general_purpose::STANDARD
                                    .decode(b64)
                                    .ok()
                                    .and_then(|png_bytes| decode_png_to_raw(&png_bytes))
                                    .map(CellValue::Bytes)
                                    .unwrap_or(CellValue::Empty)
                            } else {
                                CellValue::Empty
                            }
                        } else if let Some(serde_json::Value::String(src)) = img.get("bytes") {
                            use base64::Engine;
                            base64::engine::general_purpose::STANDARD
                                .decode(src)
                                .ok()
                                .and_then(|png_bytes| decode_png_to_raw(&png_bytes))
                                .map(CellValue::Bytes)
                                .unwrap_or(CellValue::Empty)
                        } else {
                            CellValue::Empty
                        }
                    }
                    _ => CellValue::Empty,
                };

                data_rows.push(vec![pixel_val, label_val]);
            }

            let meta = DatasetMeta {
                name: def.repo.to_string(),
                classes: def.classes.iter().map(|s| s.to_string()).collect(),
                total_samples: data_rows.len(),
                image_shape: Some(def.image_shape),
            };

            let source = VecSource {
                columns: vec![
                    ColumnMeta {
                        name: "pixels".into(),
                        kind: ColumnKind::Bytes,
                    },
                    ColumnMeta {
                        name: "label".into(),
                        kind: ColumnKind::Int,
                    },
                ],
                rows: data_rows,
            };

            Ok((meta, source))
        }
    }
}

#[cfg(feature = "dataset")]
pub use hf_provider::HuggingFaceProvider;

/// Decode PNG bytes → flat raw pixel buffer (grayscale/RGB/RGBA).
#[cfg(feature = "dataset")]
pub fn decode_png_to_raw(png_bytes: &[u8]) -> Option<Vec<u8>> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    Some(buf)
}

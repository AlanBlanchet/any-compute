//! AI/ML layer — neural network builders, dataset providers, ML taxonomy.
//!
//! Built on top of `any-compute-core`'s computational graph (`Graph`, `NodeId`)
//! and visual graph (`VisualGraph`, `Graphable`). This crate provides:
//!
//! - **Layers** — `Layer` trait + `Linear`, `Activation`, `BatchNorm`, `Residual`, `Sequential`
//! - **Networks** — `Network` builder + architecture constructors (`resnet`, `vit`, `gpt`, `mixer`)
//! - **Taxonomy** — `TaskType`, `ModelKind`, `DatasetSource` enums
//! - **Datasets** — `DatasetProvider` trait, `HuggingFaceProvider` (feature `dataset`)

pub mod dataset;
pub mod layer;
pub mod network;
pub mod taxonomy;

pub use dataset::*;
pub use layer::*;
pub use network::*;
pub use taxonomy::*;

//! ML taxonomy — task, model, and dataset domain enums.

use any_compute_core::visual::{Graphable, VisualGraph};

use super::network::{Network, gpt, mixer, resnet, vit};

/// ML task domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Detection,
    Generation,
}

impl TaskType {
    pub const ALL: [Self; 3] = [Self::Classification, Self::Detection, Self::Generation];

    pub fn label(self) -> &'static str {
        match self {
            Self::Classification => "Classification",
            Self::Detection => "Detection",
            Self::Generation => "Generation",
        }
    }

    pub fn from_tag(s: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|v| v.label().eq_ignore_ascii_case(s))
    }

    /// Emoji icon for this task type (UI display).
    pub fn icon(self) -> &'static str {
        match self {
            Self::Classification => "\u{1F3AF}",
            Self::Detection => "\u{1F50D}",
            Self::Generation => "\u{270D}",
        }
    }
}

/// Model architecture identifier with built-in network construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    ResNet,
    ViT,
    GPT,
    Mixer,
    Detr,
}

impl ModelKind {
    pub const ALL: [Self; 5] = [Self::ResNet, Self::ViT, Self::GPT, Self::Mixer, Self::Detr];

    pub fn label(self) -> &'static str {
        match self {
            Self::ResNet => "ResNet-18",
            Self::ViT => "ViT",
            Self::GPT => "GPT",
            Self::Mixer => "MLP-Mixer",
            Self::Detr => "DETR-Lite",
        }
    }

    pub fn family(self) -> &'static str {
        match self {
            Self::ResNet => "ResNet",
            Self::ViT => "Vision Transformer",
            Self::GPT => "GPT",
            Self::Mixer => "MLP-Mixer",
            Self::Detr => "Detection Transformer",
        }
    }

    pub fn task(self) -> TaskType {
        match self {
            Self::ResNet | Self::ViT | Self::Mixer => TaskType::Classification,
            Self::GPT => TaskType::Generation,
            Self::Detr => TaskType::Detection,
        }
    }

    pub fn for_task(task: TaskType) -> &'static [Self] {
        match task {
            TaskType::Classification => &[Self::ResNet, Self::ViT, Self::Mixer],
            TaskType::Detection => &[Self::Detr],
            TaskType::Generation => &[Self::GPT],
        }
    }

    pub fn from_tag(s: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|v| v.label().eq_ignore_ascii_case(s))
    }

    /// Construct the demo [`Network`] for this architecture.
    pub fn build_network(self) -> Network {
        match self {
            Self::ResNet | Self::Detr => resnet(4, 8, 2, 3),
            Self::ViT => vit(16, 64, 4, 4, 10),
            Self::GPT => gpt(256, 64, 4, 4),
            Self::Mixer => mixer(16, 64, 4, 10),
        }
    }
}

impl Graphable for ModelKind {
    fn to_graph(&self) -> VisualGraph<2> {
        self.build_network().to_visual()
    }
}

/// Dataset identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatasetSource {
    Cifar10,
    Mnist,
    TinyShakespeare,
    Synthetic,
}

impl DatasetSource {
    pub const ALL: [Self; 4] = [
        Self::Cifar10,
        Self::Mnist,
        Self::TinyShakespeare,
        Self::Synthetic,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Cifar10 => "CIFAR-10",
            Self::Mnist => "MNIST",
            Self::TinyShakespeare => "Tiny Shakespeare",
            Self::Synthetic => "Synthetic",
        }
    }

    pub fn for_task(task: TaskType) -> &'static [Self] {
        match task {
            TaskType::Classification => &[Self::Cifar10, Self::Mnist, Self::Synthetic],
            TaskType::Detection => &[Self::Synthetic],
            TaskType::Generation => &[Self::TinyShakespeare, Self::Synthetic],
        }
    }

    pub fn size_info(self) -> &'static str {
        match self {
            Self::Cifar10 => "60K images \u{2022} 10 classes \u{2022} 32\u{00D7}32",
            Self::Mnist => "70K images \u{2022} 10 digits \u{2022} 28\u{00D7}28",
            Self::TinyShakespeare => "1.1M chars \u{2022} ~40K lines",
            Self::Synthetic => "Procedural \u{2022} unlimited",
        }
    }

    pub fn from_tag(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|v| {
            let norm = v.label().to_lowercase().replace(' ', "-");
            norm == s
        })
    }
}

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A shader in its source form — not yet compiled or validated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShaderSource {
    /// WebGPU Shading Language (recommended — most portable).
    Wgsl(String),
    /// GLSL (specify version + stage).
    Glsl {
        code: String,
        stage: ShaderStage,
        version: GlslVersion,
    },
    /// Pre-compiled SPIR-V binary.
    SpirV(Vec<u8>),
}

/// GPU pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

display_enum!(ShaderStage {
    Vertex   => "vertex",
    Fragment => "fragment",
    Compute  => "compute",
});

/// GLSL version target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlslVersion {
    /// OpenGL ES 3.0 (mobile, WebGL2)
    Es300,
    /// OpenGL ES 3.1 (mobile compute)
    Es310,
    /// Desktop OpenGL 4.5
    V450,
}

// ── Compiled shader object ────────────────────────────────────────────────

/// A compiled, validated shader ready for binding and dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderObject {
    /// Human-readable label (for debugging / profiling).
    pub label: String,
    /// The stage this shader targets.
    pub stage: ShaderStage,
    /// Metadata extracted from the compiled module.
    pub metadata: ShaderMetadata,
    /// The original source format.
    pub source_format: SourceFormat,
    /// Compiled SPIR-V (if available). Backends consume this.
    #[serde(skip)]
    pub(super) spirv: Option<Vec<u8>>,
    /// Compiled WGSL (if available).
    pub(super) wgsl: Option<String>,
}

/// Which format the shader was originally written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceFormat {
    Wgsl,
    Glsl,
    SpirV,
}

/// Metadata extracted from a compiled shader module.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShaderMetadata {
    /// Entry point name.
    pub entry_point: String,
    /// Uniform / storage buffer bindings: (group, binding) → name.
    pub bindings: HashMap<(u32, u32), BindingInfo>,
    /// Workgroup size for compute shaders [x, y, z].
    pub workgroup_size: Option<[u32; 3]>,
    /// Push constant size in bytes (0 if none).
    pub push_constant_bytes: u32,
}

/// Info about a single binding slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingInfo {
    pub name: String,
    pub kind: BindingKind,
    /// Size in bytes (0 if runtime-sized array).
    pub size_bytes: u32,
}

/// Type of binding resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingKind {
    UniformBuffer,
    StorageBuffer,
    ReadOnlyStorageBuffer,
    Sampler,
    Texture2D,
    Texture3D,
    StorageTexture,
}

// ── Shader compiler (requires `shader` feature) ──────────────────────────

/// Error type for shader compilation.
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("validation error: {0}")]
    Validation(String),
    #[error("cross-compilation error: {0}")]
    CrossCompile(String),
    #[error("shader feature not enabled — add `shader` to Cargo features")]
    FeatureDisabled,
}

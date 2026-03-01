//! Shader compilation and management — WGSL, GLSL, SPIR-V cross-compilation.
//!
//! This module provides a unified shader pipeline:
//! 1. Write shaders in any supported language (WGSL, GLSL, SPIR-V)
//! 2. Cross-compile to any target via [`naga`] (behind `shader` feature)
//! 3. Cache compiled artifacts for fast re-use
//!
//! ## Shader Object Model
//!
//! A [`ShaderObject`] is a compiled, inspectable shader ready for dispatch.
//! It carries metadata about inputs/outputs, uniforms, and workgroup size
//! so the engine can validate bindings at creation time rather than at dispatch.
//!
//! ## Pipeline
//!
//! ```text
//! ShaderSource (WGSL / GLSL / SPIR-V bytes)
//!     │
//!     ▼
//! ShaderCompiler::compile()          ◄── validates + cross-compiles
//!     │
//!     ▼
//! ShaderObject { module, metadata }  ◄── cached, inspectable
//!     │
//!     ▼
//! ShaderObject::to_*()               ◄── emit WGSL / GLSL / SPIR-V
//! ```
//!
//! ## Without the `shader` feature
//!
//! When `shader` is not enabled, only [`ShaderSource`] and placeholder types
//! are available. Compilation requires the feature.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Source representation ─────────────────────────────────────────────────

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

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vertex => write!(f, "vertex"),
            Self::Fragment => write!(f, "fragment"),
            Self::Compute => write!(f, "compute"),
        }
    }
}

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
    spirv: Option<Vec<u8>>,
    /// Compiled WGSL (if available).
    wgsl: Option<String>,
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

/// Shader compiler — validates and cross-compiles between WGSL, GLSL, SPIR-V.
///
/// Requires the `shader` cargo feature (which pulls in `naga`).
pub struct ShaderCompiler;

impl ShaderCompiler {
    /// Compile a shader source into a [`ShaderObject`].
    ///
    /// This validates the shader, extracts metadata, and prepares it for
    /// cross-compilation to any target format.
    #[cfg(feature = "shader")]
    pub fn compile(source: &ShaderSource, label: &str) -> Result<ShaderObject, ShaderError> {
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let (module, source_format) = match source {
            ShaderSource::Wgsl(code) => {
                let module = naga::front::wgsl::parse_str(code)
                    .map_err(|e| ShaderError::Parse(format!("{e}")))?;
                (module, SourceFormat::Wgsl)
            }
            ShaderSource::Glsl {
                code,
                stage,
                version: _,
            } => {
                let naga_stage = match stage {
                    ShaderStage::Vertex => naga::ShaderStage::Vertex,
                    ShaderStage::Fragment => naga::ShaderStage::Fragment,
                    ShaderStage::Compute => naga::ShaderStage::Compute,
                };
                let mut opts = naga::front::glsl::Options::from(naga_stage);
                opts.defines.clear();
                let module = naga::front::glsl::Frontend::default()
                    .parse(&opts, code)
                    .map_err(|e| ShaderError::Parse(format!("{e:?}")))?;
                (module, SourceFormat::Glsl)
            }
            ShaderSource::SpirV(bytes) => {
                let opts = naga::front::spv::Options::default();
                let module = naga::front::spv::parse_u8_slice(bytes, &opts)
                    .map_err(|e| ShaderError::Parse(format!("{e}")))?;
                (module, SourceFormat::SpirV)
            }
        };

        // Validate
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .map_err(|e| ShaderError::Validation(format!("{e}")))?;

        // Extract metadata
        let metadata = extract_metadata(&module);

        // Cross-compile to WGSL
        let wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .map_err(|e| ShaderError::CrossCompile(format!("{e}")))?;

        // Cross-compile to SPIR-V
        let spirv_opts = naga::back::spv::Options::default();
        let pipeline_opts = naga::back::spv::PipelineOptions {
            shader_stage: module
                .entry_points
                .first()
                .map(|ep| ep.stage)
                .unwrap_or(naga::ShaderStage::Compute),
            entry_point: metadata.entry_point.clone(),
        };
        let spirv_words =
            naga::back::spv::write_vec(&module, &info, &spirv_opts, Some(&pipeline_opts))
                .map_err(|e| ShaderError::CrossCompile(format!("{e}")))?;
        let spirv_bytes: Vec<u8> = spirv_words.iter().flat_map(|w| w.to_le_bytes()).collect();

        Ok(ShaderObject {
            label: label.to_string(),
            stage: metadata
                .workgroup_size
                .map(|_| ShaderStage::Compute)
                .unwrap_or(ShaderStage::Vertex),
            metadata,
            source_format,
            spirv: Some(spirv_bytes),
            wgsl: Some(wgsl),
        })
    }

    /// Stub when `shader` feature is not enabled.
    #[cfg(not(feature = "shader"))]
    pub fn compile(_source: &ShaderSource, _label: &str) -> Result<ShaderObject, ShaderError> {
        Err(ShaderError::FeatureDisabled)
    }
}

/// Extract metadata from a naga module.
#[cfg(feature = "shader")]
fn extract_metadata(module: &naga::Module) -> ShaderMetadata {
    let entry_point = module
        .entry_points
        .first()
        .map(|ep| ep.name.clone())
        .unwrap_or_default();

    let workgroup_size = module.entry_points.first().and_then(|ep| {
        if ep.stage == naga::ShaderStage::Compute {
            Some(ep.workgroup_size)
        } else {
            None
        }
    });

    let mut bindings = HashMap::new();
    for (_, var) in module.global_variables.iter() {
        if let Some(ref binding) = var.binding {
            let kind = match var.space {
                naga::AddressSpace::Uniform => BindingKind::UniformBuffer,
                naga::AddressSpace::Storage { access } => {
                    if access.contains(naga::StorageAccess::STORE) {
                        BindingKind::StorageBuffer
                    } else {
                        BindingKind::ReadOnlyStorageBuffer
                    }
                }
                _ => continue,
            };
            bindings.insert(
                (binding.group, binding.binding),
                BindingInfo {
                    name: var.name.clone().unwrap_or_default(),
                    kind,
                    size_bytes: 0, // Would need type-size resolution
                },
            );
        }
    }

    ShaderMetadata {
        entry_point,
        bindings,
        workgroup_size,
        push_constant_bytes: 0,
    }
}

impl ShaderObject {
    /// Get the compiled WGSL source (if available).
    pub fn to_wgsl(&self) -> Option<&str> {
        self.wgsl.as_deref()
    }

    /// Get the compiled SPIR-V bytes (if available).
    pub fn to_spirv(&self) -> Option<&[u8]> {
        self.spirv.as_deref()
    }

    /// Cross-compile to GLSL (requires `shader` feature).
    #[cfg(feature = "shader")]
    pub fn to_glsl(&self, version: GlslVersion) -> Result<String, ShaderError> {
        // Re-parse from WGSL then emit GLSL
        let wgsl = self.wgsl.as_deref().ok_or(ShaderError::CrossCompile(
            "no WGSL representation available".into(),
        ))?;
        let module = naga::front::wgsl::parse_str(wgsl)
            .map_err(|e| ShaderError::CrossCompile(format!("{e}")))?;

        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .map_err(|e| ShaderError::CrossCompile(format!("{e}")))?;

        let naga_version = match version {
            GlslVersion::Es300 => naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: false,
            },
            GlslVersion::Es310 => naga::back::glsl::Version::Embedded {
                version: 310,
                is_webgl: false,
            },
            GlslVersion::V450 => naga::back::glsl::Version::Desktop(450),
        };

        let ep = module
            .entry_points
            .first()
            .ok_or_else(|| ShaderError::CrossCompile("no entry points".into()))?;

        let opts = naga::back::glsl::Options {
            version: naga_version,
            writer_flags: naga::back::glsl::WriterFlags::empty(),
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: true,
        };

        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: ep.stage,
            entry_point: ep.name.clone(),
            multiview: None,
        };

        let mut output = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &opts,
            &pipeline,
            Default::default(),
        )
        .map_err(|e| ShaderError::CrossCompile(format!("{e}")))?;
        writer
            .write()
            .map_err(|e| ShaderError::CrossCompile(format!("{e}")))?;

        Ok(output)
    }
}

// ── Built-in shader templates ─────────────────────────────────────────────

/// Common compute shader templates that users can customize.
///
/// Shader source lives in `crates/core/shaders/*.wgsl` — loaded at compile time
/// via `include_str!`. Parameterized templates use `{{PLACEHOLDER}}` markers
/// that are replaced at runtime.
pub mod templates {
    use super::*;

    /// Default workgroup size for 1D dispatches (map, reduce).
    pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

    /// Default tile dimension for 2D dispatches (GEMM). Shared memory = TILE^2.
    pub const DEFAULT_TILE_SIZE: u32 = 16;

    const MAP_TEMPLATE: &str = include_str!("../shaders/map.wgsl");
    const REDUCE_TEMPLATE: &str = include_str!("../shaders/reduce.wgsl");
    const GEMM_TEMPLATE: &str = include_str!("../shaders/gemm.wgsl");

    /// Replace `{{KEY}}` placeholders in a template with concrete values.
    fn instantiate(template: &str, vars: &[(&str, &str)]) -> String {
        let mut out = template.to_string();
        for &(key, val) in vars {
            out = out.replace(&format!("{{{{{key}}}}}"), val);
        }
        out
    }

    /// A simple element-wise map shader in WGSL.
    pub fn map_shader(body: &str) -> ShaderSource {
        let wg = DEFAULT_WORKGROUP_SIZE.to_string();
        ShaderSource::Wgsl(instantiate(MAP_TEMPLATE, &[
            ("WORKGROUP_SIZE", &wg),
            ("BODY", body),
        ]))
    }

    /// A reduction shader in WGSL (workgroup-level reduce).
    pub fn reduce_shader(op: &str) -> ShaderSource {
        let wg = DEFAULT_WORKGROUP_SIZE.to_string();
        let half = (DEFAULT_WORKGROUP_SIZE / 2).to_string();
        ShaderSource::Wgsl(instantiate(REDUCE_TEMPLATE, &[
            ("WORKGROUP_SIZE", &wg),
            ("HALF_WORKGROUP", &half),
            ("OP", op),
        ]))
    }

    /// A matrix multiply shader in WGSL (tiled, shared memory).
    pub fn gemm_shader() -> ShaderSource {
        let tile = DEFAULT_TILE_SIZE.to_string();
        let elems = (DEFAULT_TILE_SIZE * DEFAULT_TILE_SIZE).to_string();
        ShaderSource::Wgsl(instantiate(GEMM_TEMPLATE, &[
            ("TILE_SIZE", &tile),
            ("TILE_ELEMENTS", &elems),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_source_wgsl_roundtrip() {
        let src = ShaderSource::Wgsl("fn main() {}".into());
        let json = serde_json::to_string(&src).unwrap();
        let back: ShaderSource = serde_json::from_str(&json).unwrap();
        match back {
            ShaderSource::Wgsl(code) => assert!(code.contains("main")),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn shader_source_glsl_roundtrip() {
        let src = ShaderSource::Glsl {
            code: "void main() {}".into(),
            stage: ShaderStage::Compute,
            version: GlslVersion::V450,
        };
        let json = serde_json::to_string(&src).unwrap();
        let back: ShaderSource = serde_json::from_str(&json).unwrap();
        match back {
            ShaderSource::Glsl { stage, .. } => assert_eq!(stage, ShaderStage::Compute),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn template_map_shader() {
        let src = templates::map_shader("v * 2.0");
        match src {
            ShaderSource::Wgsl(code) => {
                assert!(code.contains("@compute"));
                assert!(code.contains("v * 2.0"));
                let wg = templates::DEFAULT_WORKGROUP_SIZE.to_string();
                assert!(code.contains(&format!("@workgroup_size({wg})")));
            }
            _ => panic!("expected WGSL"),
        }
    }

    #[test]
    fn template_reduce_shader() {
        let src = templates::reduce_shader("shared[lid.x] + shared[lid.x + stride]");
        match src {
            ShaderSource::Wgsl(code) => {
                assert!(code.contains("workgroupBarrier"));
                assert!(code.contains("shared[lid.x] + shared[lid.x + stride]"));
                // No unresolved placeholders
                assert!(!code.contains("{{"));
            }
            _ => panic!("expected WGSL"),
        }
    }

    #[test]
    fn template_gemm_shader() {
        match templates::gemm_shader() {
            ShaderSource::Wgsl(code) => {
                assert!(code.contains("tileA"));
                assert!(code.contains("workgroupBarrier"));
                // Tile size should be resolved
                let tile = templates::DEFAULT_TILE_SIZE.to_string();
                assert!(code.contains(&format!("const TILE: u32 = {tile}u")));
                // No unresolved placeholders
                assert!(!code.contains("{{"));
            }
            _ => panic!("expected WGSL"),
        }
    }

    #[test]
    fn shader_stage_display() {
        assert_eq!(ShaderStage::Compute.to_string(), "compute");
        assert_eq!(ShaderStage::Vertex.to_string(), "vertex");
        assert_eq!(ShaderStage::Fragment.to_string(), "fragment");
    }

    #[test]
    fn metadata_default() {
        let m = ShaderMetadata::default();
        assert!(m.entry_point.is_empty());
        assert!(m.bindings.is_empty());
        assert!(m.workgroup_size.is_none());
    }

    #[cfg(not(feature = "shader"))]
    #[test]
    fn compile_disabled_without_feature() {
        let src = ShaderSource::Wgsl("fn main() {}".into());
        let result = ShaderCompiler::compile(&src, "test");
        assert!(matches!(result, Err(ShaderError::FeatureDisabled)));
    }
}

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



mod types;
mod compiler;

pub use types::*;
pub use compiler::*;

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

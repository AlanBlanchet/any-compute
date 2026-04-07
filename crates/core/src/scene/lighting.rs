use crate::layout::{V, V3};
use crate::render::Color;

// ═══════════════════════════════════════════════════════════════════════════
// ── Light ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Light source.
#[derive(Debug, Clone)]
pub enum Light {
    /// Direction toward the light (infinite distance).
    Directional {
        direction: V3,
        color: V3,
        intensity: f64,
    },
    /// Point light with position and falloff.
    Point {
        position: V3,
        color: V3,
        intensity: f64,
        range: f64,
    },
    /// Ambient light (uniform everywhere).
    Ambient { color: V3, intensity: f64 },
}

impl Light {
    /// Compute this light's RGB contribution to a face with the given normal.
    /// Returns (r, g, b) in 0..1 range.
    pub fn contribution(&self, normal: V3) -> V3 {
        match self {
            Light::Directional {
                direction,
                color,
                intensity,
            } => {
                let ndl = normal.dot(*direction * -1.0).max(0.0);
                *color * (*intensity * ndl)
            }
            Light::Ambient { color, intensity } => *color * *intensity,
            Light::Point {
                color, intensity, ..
            } => *color * (*intensity * 0.1),
        }
    }
}

/// Shade a face: accumulate light contributions and modulate a base color.
/// Prefer [`Scene::shade_face`] which binds lights automatically.
pub(crate) fn shade_face(normal: V3, base: Color, lights: &[Light]) -> Color {
    let mut c = V3::ZERO;
    for light in lights {
        c = c + light.contribution(normal);
    }
    Color::rgb(
        (base.r as f64 * c.0[0].min(1.0)) as u8,
        (base.g as f64 * c.0[1].min(1.0)) as u8,
        (base.b as f64 * c.0[2].min(1.0)) as u8,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Material ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// PBR-style material.
#[derive(Debug, Clone)]
pub struct Material {
    /// Base color (linear RGB, 0..1).
    pub albedo: V3,
    /// 0 = dielectric, 1 = metal.
    pub metallic: f64,
    /// 0 = mirror, 1 = rough.
    pub roughness: f64,
    /// Emission color (linear RGB, 0..1). Zero = no emission.
    pub emissive: V3,
    /// Index of refraction (1.0 = air, 1.5 = glass).
    pub ior: f64,
    /// Opacity (1.0 = opaque, 0.0 = fully transparent).
    pub alpha: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: V([0.8, 0.8, 0.8]),
            metallic: 0.0,
            roughness: 0.5,
            emissive: V3::ZERO,
            ior: 1.5,
            alpha: 1.0,
        }
    }
}


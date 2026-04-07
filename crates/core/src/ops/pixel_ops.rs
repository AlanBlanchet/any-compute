use super::traits::colors_from_rgba;
use crate::render::{Color, PixelBuffer};

// ═══════════════════════════════════════════════════════════════════════════
// ── PixelBuffer extra utilities ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl PixelBuffer {
    /// Convert to raw RGBA bytes.
    pub fn to_rgba(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|c| [c.r, c.g, c.b, c.a])
            .collect()
    }

    /// Encode to PNG bytes.
    #[cfg(feature = "png")]
    pub fn to_png(&self) -> Vec<u8> {
        encode_png(&self.to_rgba(), self.width, self.height)
    }

    /// Save to a PNG file.
    #[cfg(feature = "png")]
    pub fn save_png(&self, path: impl AsRef<std::path::Path>) {
        let bytes = self.to_png();
        std::fs::write(path, bytes).expect("failed to write PNG");
    }

    /// Create from raw RGBA bytes.
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        Self {
            width,
            height,
            pixels: colors_from_rgba(rgba),
            clear: Color::BLACK,
        }
    }

    /// Downscale by an integer factor (box filter / average).
    pub fn downscale(&self, factor: u32) -> Self {
        if factor <= 1 {
            return self.clone();
        }
        let nw = self.width / factor;
        let nh = self.height / factor;
        let mut out = Self::new(nw, nh, self.clear);
        let area = (factor * factor) as f64;
        for y in 0..nh {
            for x in 0..nw {
                let (mut r, mut g, mut b, mut a) = (0u32, 0u32, 0u32, 0u32);
                for dy in 0..factor {
                    for dx in 0..factor {
                        let c = self.pixel(x * factor + dx, y * factor + dy);
                        r += c.r as u32;
                        g += c.g as u32;
                        b += c.b as u32;
                        a += c.a as u32;
                    }
                }
                out.pixels[(y * nw + x) as usize] = Color::rgba(
                    (r as f64 / area) as u8,
                    (g as f64 / area) as u8,
                    (b as f64 / area) as u8,
                    (a as f64 / area) as u8,
                );
            }
        }
        out
    }

    /// Crop a sub-region.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Self {
        let mut out = Self::new(w, h, self.clear);
        for dy in 0..h {
            for dx in 0..w {
                out.pixels[(dy * w + dx) as usize] = self.pixel(x + dx, y + dy);
            }
        }
        out
    }

    /// Flip horizontally.
    pub fn flip_h(&self) -> Self {
        let mut out = self.clone();
        for y in 0..self.height {
            for x in 0..self.width {
                out.pixels[(y * self.width + x) as usize] = self.pixel(self.width - 1 - x, y);
            }
        }
        out
    }

    /// Flip vertically.
    pub fn flip_v(&self) -> Self {
        let mut out = self.clone();
        for y in 0..self.height {
            for x in 0..self.width {
                out.pixels[(y * self.width + x) as usize] = self.pixel(x, self.height - 1 - y);
            }
        }
        out
    }

    /// Count pixels matching a specific color (exact match).
    pub fn count_color(&self, color: Color) -> usize {
        self.pixels.iter().filter(|&&c| c == color).count()
    }

    /// Dominant color (most frequent).
    pub fn dominant_color(&self) -> Color {
        let mut freq = std::collections::HashMap::<u32, usize>::new();
        for c in &self.pixels {
            *freq.entry(u32::from(*c)).or_default() += 1;
        }
        freq.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(hex, _)| Color::from(hex))
            .unwrap_or(Color::BLACK)
    }

    /// Average color across all pixels.
    pub fn average_color(&self) -> Color {
        if self.pixels.is_empty() {
            return Color::BLACK;
        }
        let (mut r, mut g, mut b, mut a) = (0u64, 0u64, 0u64, 0u64);
        for c in &self.pixels {
            r += c.r as u64;
            g += c.g as u64;
            b += c.b as u64;
            a += c.a as u64;
        }
        let n = self.pixels.len() as u64;
        Color::rgba((r / n) as u8, (g / n) as u8, (b / n) as u8, (a / n) as u8)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── PNG encoding ────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Encode RGBA bytes to PNG format.
#[cfg(feature = "png")]
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(std::io::Cursor::new(&mut out), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("PNG header write failed");
        writer
            .write_image_data(rgba)
            .expect("PNG data write failed");
    }
    out
}


// ═══════════════════════════════════════════════════════════════════════════
// ── PixelBuffer color space conversions ─────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl PixelBuffer {
    /// Convert to grayscale using luminance weights (ITU-R BT.709).
    pub fn to_grayscale(&self) -> Self {
        let mut out = self.clone();
        for px in &mut out.pixels {
            let lum = (0.2126 * px.r as f64 + 0.7152 * px.g as f64 + 0.0722 * px.b as f64) as u8;
            *px = Color::rgba(lum, lum, lum, px.a);
        }
        out
    }

    /// Invert colors (255 - channel).
    pub fn invert(&self) -> Self {
        let mut out = self.clone();
        for px in &mut out.pixels {
            *px = Color::rgba(255 - px.r, 255 - px.g, 255 - px.b, px.a);
        }
        out
    }

    /// Adjust brightness by a factor (1.0 = unchanged, >1 = brighter).
    pub fn brightness(&self, factor: f64) -> Self {
        let mut out = self.clone();
        for px in &mut out.pixels {
            *px = Color::rgba(
                (px.r as f64 * factor).clamp(0.0, 255.0) as u8,
                (px.g as f64 * factor).clamp(0.0, 255.0) as u8,
                (px.b as f64 * factor).clamp(0.0, 255.0) as u8,
                px.a,
            );
        }
        out
    }

    /// Alpha-composite another buffer on top at position (x, y).
    pub fn composite(&self, other: &Self, ox: u32, oy: u32) -> Self {
        let mut out = self.clone();
        for y in 0..other.height {
            for x in 0..other.width {
                let dx = ox + x;
                let dy = oy + y;
                if dx < out.width && dy < out.height {
                    let src = other.pixel(x, y);
                    if src.a == 0 {
                        continue;
                    }
                    let dst = out.pixel(dx, dy);
                    let sa = src.a as f64 / 255.0;
                    let da = dst.a as f64 / 255.0;
                    let oa = sa + da * (1.0 - sa);
                    if oa < f64::EPSILON {
                        continue;
                    }
                    let blend = |s: u8, d: u8| -> u8 {
                        ((s as f64 * sa + d as f64 * da * (1.0 - sa)) / oa).round() as u8
                    };
                    out.pixels[(dy * out.width + dx) as usize] = Color::rgba(
                        blend(src.r, dst.r),
                        blend(src.g, dst.g),
                        blend(src.b, dst.b),
                        (oa * 255.0) as u8,
                    );
                }
            }
        }
        out
    }

    /// Compare with another PixelBuffer, returning per-pixel absolute difference image.
    pub fn diff_image(&self, other: &Self) -> Self {
        let w = self.width.min(other.width);
        let h = self.height.min(other.height);
        let mut out = Self::new(w, h, Color::BLACK);
        for y in 0..h {
            for x in 0..w {
                let a = self.pixel(x, y);
                let b = other.pixel(x, y);
                out.pixels[(y * w + x) as usize] = Color::rgba(
                    (a.r as i16 - b.r as i16).unsigned_abs() as u8,
                    (a.g as i16 - b.g as i16).unsigned_abs() as u8,
                    (a.b as i16 - b.b as i16).unsigned_abs() as u8,
                    255,
                );
            }
        }
        out
    }

    /// Total pixel difference count (pixels where any channel differs by > threshold).
    pub fn diff_count(&self, other: &Self, threshold: u8) -> usize {
        let w = self.width.min(other.width);
        let h = self.height.min(other.height);
        let t = threshold as i16;
        let mut count = 0;
        for y in 0..h {
            for x in 0..w {
                let a = self.pixel(x, y);
                let b = other.pixel(x, y);
                if (a.r as i16 - b.r as i16).abs() > t
                    || (a.g as i16 - b.g as i16).abs() > t
                    || (a.b as i16 - b.b as i16).abs() > t
                {
                    count += 1;
                }
            }
        }
        count
    }

    /// Check that a rectangular region is uniformly one color (within tolerance).
    pub fn region_uniform(&self, x: u32, y: u32, w: u32, h: u32, tolerance: u8) -> bool {
        let first = self.pixel(x, y);
        let t = tolerance as i16;
        for dy in 0..h {
            for dx in 0..w {
                let c = self.pixel(x + dx, y + dy);
                if (c.r as i16 - first.r as i16).abs() > t
                    || (c.g as i16 - first.g as i16).abs() > t
                    || (c.b as i16 - first.b as i16).abs() > t
                {
                    return false;
                }
            }
        }
        true
    }
}


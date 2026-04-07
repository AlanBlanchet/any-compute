use crate::render::{Color, PixelBuffer, RenderList, Renderable};

#[cfg(feature = "png")]
use super::pixel_ops::encode_png;

// ═══════════════════════════════════════════════════════════════════════════
// ── Capture — snapshot Renderable → pixels ──────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Snapshot a renderable object into a pixel buffer.
///
/// Any `Renderable<()>` gets this for free — the default implementation
/// renders to a CPU `PixelBuffer` using the SDF rasterizer.
///
/// ## Generic → specific pattern
///
/// - `capture()` — generic entry (returns PixelBuffer)
/// - `capture_rgba()` — specific format (raw bytes)
/// - `capture_png()` — specific format (PNG encoded)
/// - `save()` — generic save (auto-selects by extension, defaults to PNG)
/// - `save_png()` — specific save
pub trait Capture: Renderable<()> {
    /// Render to a CPU pixel buffer at the given dimensions.
    fn capture(&self, width: u32, height: u32) -> PixelBuffer {
        let mut list = RenderList::default();
        self.render(&mut list, &());
        let mut buf = PixelBuffer::new(width, height, Color::BLACK);
        buf.paint(&list);
        buf
    }

    /// Render to raw RGBA bytes (4 bytes per pixel, row-major).
    fn capture_rgba(&self, width: u32, height: u32) -> Vec<u8> {
        let buf = self.capture(width, height);
        buf.pixels
            .iter()
            .flat_map(|c| [c.r, c.g, c.b, c.a])
            .collect()
    }

    /// Render to PNG bytes (in-memory).
    #[cfg(feature = "png")]
    fn capture_png(&self, width: u32, height: u32) -> Vec<u8> {
        let rgba = self.capture_rgba(width, height);
        encode_png(&rgba, width, height)
    }

    /// Generic save — auto-selects format by file extension.
    /// Defaults to PNG. Override for type-specific formats.
    #[cfg(feature = "png")]
    fn save(&self, path: impl AsRef<std::path::Path>, width: u32, height: u32) {
        self.save_png(path, width, height);
    }

    /// Render and save to a PNG file.
    #[cfg(feature = "png")]
    fn save_png(&self, path: impl AsRef<std::path::Path>, width: u32, height: u32) {
        let bytes = self.capture_png(width, height);
        std::fs::write(path, bytes).expect("failed to write PNG");
    }
}

// Blanket: every Renderable<()> is Capture.
impl<T: Renderable<()>> Capture for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── Record — multi-frame capture ────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A recorded sequence of frames, captured from any `Renderable`.
pub struct Recording {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<Vec<u8>>,
}

impl Recording {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frames: Vec::new(),
        }
    }

    /// Capture a single frame from a renderable.
    pub fn frame(&mut self, obj: &impl Capture) {
        self.frames.push(obj.capture_rgba(self.width, self.height));
    }

    /// Number of captured frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Save all frames as numbered PNGs in a directory.
    #[cfg(feature = "png")]
    pub fn save_frames(&self, dir: impl AsRef<std::path::Path>) {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).expect("failed to create directory");
        for (i, rgba) in self.frames.iter().enumerate() {
            let path = dir.join(format!("frame_{i:04}.png"));
            let bytes = encode_png(rgba, self.width, self.height);
            std::fs::write(path, bytes).expect("failed to write frame PNG");
        }
    }

    /// Get a specific frame as a PixelBuffer.
    pub fn pixel_buffer(&self, frame_idx: usize) -> PixelBuffer {
        PixelBuffer::from_rgba(self.width, self.height, &self.frames[frame_idx])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Export — format-specific serialization ───────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Export a type to various format strings.
///
/// `export()` is the generic entry point — it auto-selects the best format
/// (JSON by default). Override it to customize format selection.
/// `to_json()` / `to_csv()` are the format-specific specializations.
pub trait Export {
    /// Generic export — auto-selects the best format.
    /// Override this for type-specific default format selection.
    fn export(&self) -> String {
        self.to_json()
    }
    /// Serialize to JSON string (if applicable).
    fn to_json(&self) -> String {
        String::new()
    }
    /// Serialize to CSV string (if applicable).
    fn to_csv(&self) -> String {
        String::new()
    }
}

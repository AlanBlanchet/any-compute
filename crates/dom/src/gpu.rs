//! Reusable WGPU renderer — instanced SDF rects + glyphon text.
//!
//! Supports both windowed (`init`) and headless (`init_headless`) modes.
//! The shader is loaded from `shaders/rect.wgsl` via `include_str!`.

use crate::theme;
use any_compute_core::layout::Point;
use any_compute_core::render::{Color, Primitive, RenderList};
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache as GlyphCache, Family, FontSystem, Metrics, Resolution,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use pollster::block_on;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use winit::window::Window;

/// SDF rounded-rect + border shader, loaded from file (no inline WGSL).
const SHADER_CODE: &str = include_str!("../shaders/rect.wgsl");

/// Default line-height multiplier for glyphon text shaping.
const GLYPHON_LINE_HEIGHT: f32 = 1.2;

/// Glyphon text-area top offset as a fraction of font_size.
/// Paired with `TEXT_BASELINE_RATIO` (0.85) in the DOM — the 0.05 delta
/// centers the glyphon rendering within the DOM line-height box.
const GLYPHON_TEXT_TOP_RATIO: f32 = 0.8;

/// Maximum SDF rect instances per frame.
const MAX_INSTANCES: usize = 50_000;

/// Width ceiling for text measurement (effectively unbounded single-line).
const MEASURE_MAX_WIDTH: f32 = 10_000.0;

// ── sRGB → linear conversion ────────────────────────────────────────────────

/// Convert a single sRGB 0-255 channel to linear 0.0-1.0.
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert an RGBA `Color` (sRGB) into `[f32; 4]` in linear space.
#[inline]
fn color_linear(c: Color) -> [f32; 4] {
    [
        srgb_to_linear(c.r),
        srgb_to_linear(c.g),
        srgb_to_linear(c.b),
        c.a as f32 / 255.0,
    ]
}

// ── GPU types ───────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub bounds: [f32; 4],
    pub color: [f32; 4],
    pub params: [f32; 4],
    pub border_color: [f32; 4],
    pub border_widths: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniforms {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

/// A contiguous range of rect instances sharing the same scissor clip.
#[derive(Clone)]
struct DrawBatch {
    start: u32,
    count: u32,
    /// Scissor rect in pixels (x, y, w, h). `None` = full viewport.
    scissor: Option<[u32; 4]>,
}

/// Intersect two clip rects (x, y, w, h). Returns the overlapping region.
fn intersect_clip(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = (a[0] + a[2]).min(b[0] + b[2]);
    let y1 = (a[1] + a[3]).min(b[1] + b[3]);
    if x1 <= x0 || y1 <= y0 {
        [0, 0, 0, 0]
    } else {
        [x0, y0, x1 - x0, y1 - y0]
    }
}

// ── Text cache helpers ──────────────────────────────────────────────────────

/// Compute a cache key for shaped text: hash of content + font-size bits.
fn text_cache_key(text: &str, font_size: f32) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    text.hash(&mut h);
    font_size.to_bits().hash(&mut h);
    h.finish()
}

/// Shape text using glyphon (free function — usable without &mut Gpu).
fn shape_text_into(
    font_system: &mut FontSystem,
    text: &str,
    font_size: f32,
    width: Option<f32>,
    height: Option<f32>,
) -> GlyphBuffer {
    let mut buf = GlyphBuffer::new(
        font_system,
        Metrics::new(font_size, font_size * GLYPHON_LINE_HEIGHT),
    );
    buf.set_size(font_system, width, height);
    buf.set_text(
        font_system,
        text,
        Attrs::new().family(Family::SansSerif),
        Shaping::Advanced,
    );
    buf.shape_until_scroll(font_system, false);
    buf
}

// ── Gpu renderer ────────────────────────────────────────────────────────────

pub struct Gpu {
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    ub: wgpu::Buffer,
    bg: wgpu::BindGroup,
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_viewport: Viewport,
    text_renderer: TextRenderer,
    /// Reusable instance buffer staging — cleared each frame, avoids allocation.
    rect_instances: Vec<InstanceData>,
    /// Text shaping cache: hash(content + font_size) → shaped GlyphBuffer.
    /// Avoids re-shaping unchanged text every frame (the #1 cost).
    text_cache: HashMap<u64, GlyphBuffer>,
    /// Clear color (defaults to Catppuccin Mocha BG).
    pub clear: Color,
}

impl Gpu {
    /// Shared setup: shader, buffers, pipeline, text renderer.
    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: Option<wgpu::Surface<'static>>,
        fmt: wgpu::TextureFormat,
        w: u32,
        h: u32,
    ) -> Self {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        if let Some(ref s) = surface {
            s.configure(&device, &config);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui"),
            source: wgpu::ShaderSource::Wgsl(SHADER_CODE.into()),
        });
        use wgpu::util::DeviceExt;
        let verts: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ib"),
            size: (std::mem::size_of::<InstanceData>() * MAX_INSTANCES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u = GpuUniforms {
            screen_size: [w as f32, h as f32],
            _pad: [0.0; 2],
        };
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ub"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("bgl"),
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            }],
            label: Some("bg"),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rp"),
            layout: Some(&pl),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstanceData>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            1 => Float32x4,
                            2 => Float32x4,
                            3 => Float32x4,
                            4 => Float32x4,
                            5 => Float32x4,
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = GlyphCache::new(&device);
        let text_viewport = Viewport::new(&device, &glyph_cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyph_cache, fmt);
        let text_renderer = TextRenderer::new(&mut text_atlas, &device, Default::default(), None);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            vb,
            ib,
            ub,
            bg,
            font_system,
            swash_cache,
            text_atlas,
            text_viewport,
            text_renderer,
            rect_instances: Vec::with_capacity(4096),
            text_cache: HashMap::new(),
            clear: theme::BG,
        }
    }

    /// Create a GPU renderer backed by a visible window surface.
    pub fn init(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = inst.create_surface(window.clone()).unwrap();
        let adapter = block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        Self::build(device, queue, Some(surface), fmt, size.width, size.height)
    }

    /// Create a GPU renderer without a window — capture-only.
    pub fn init_headless(w: u32, h: u32) -> Self {
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
        Self::build(device, queue, None, fmt, w, h)
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        if let Some(ref s) = self.surface {
            s.configure(&self.device, &self.config);
        }
        let u = GpuUniforms {
            screen_size: [w as f32, h as f32],
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.ub, 0, bytemuck::bytes_of(&u));
    }

    // ── Shared helpers ─────────────────────────────────────────────────────

    fn prepare(&mut self, list: &RenderList) -> Vec<DrawBatch> {
        let (vw, vh) = (self.config.width, self.config.height);

        // ── Rect instances + clip batches ───────────────────────────────
        self.rect_instances.clear();
        let mut batches: Vec<DrawBatch> = Vec::new();
        let mut clip_stack: Vec<[u32; 4]> = Vec::new();
        let mut cur_scissor: Option<[u32; 4]> = None;
        let mut batch_start: u32 = 0;

        // Flush current batch when scissor changes.
        macro_rules! flush_batch {
            () => {
                let n = self.rect_instances.len() as u32;
                if n > batch_start {
                    batches.push(DrawBatch {
                        start: batch_start,
                        count: n - batch_start,
                        scissor: cur_scissor,
                    });
                    batch_start = n;
                }
            };
        }

        for p in &list.primitives {
            match p {
                Primitive::PushClip { bounds } => {
                    flush_batch!();
                    let new = [
                        bounds.origin.x.max(0.0) as u32,
                        bounds.origin.y.max(0.0) as u32,
                        (bounds.size.w().ceil() as u32).min(vw),
                        (bounds.size.h().ceil() as u32).min(vh),
                    ];
                    let clipped = match cur_scissor {
                        Some(prev) => intersect_clip(prev, new),
                        None => new,
                    };
                    clip_stack.push(cur_scissor.unwrap_or([0, 0, vw, vh]));
                    cur_scissor = Some(clipped);
                }
                Primitive::PopClip => {
                    flush_batch!();
                    cur_scissor = clip_stack.pop().map(|c| {
                        if c == [0, 0, vw, vh] {
                            return c;
                        }
                        c
                    });
                    // Restore to None (full viewport) when stack is empty or was full viewport.
                    if cur_scissor == Some([0, 0, vw, vh]) {
                        cur_scissor = None;
                    }
                }
                Primitive::Rect {
                    bounds,
                    fill,
                    border,
                    corner_radius,
                } => {
                    let (bw, bc) = border
                        .map(|b| {
                            (
                                [b.top as f32, b.right as f32, b.bottom as f32, b.left as f32],
                                color_linear(b.color),
                            )
                        })
                        .unwrap_or(([0.0; 4], [0.0; 4]));
                    self.rect_instances.push(InstanceData {
                        bounds: [
                            bounds.origin.x as f32,
                            bounds.origin.y as f32,
                            bounds.size.w() as f32,
                            bounds.size.h() as f32,
                        ],
                        color: color_linear(*fill),
                        params: [*corner_radius as f32, 0.0, 0.0, 0.0],
                        border_color: bc,
                        border_widths: bw,
                    });
                }
                Primitive::Line {
                    from,
                    to,
                    stroke,
                    width,
                } => {
                    let dx = to.x - from.x;
                    let dy = to.y - from.y;
                    let length = (dx * dx + dy * dy).sqrt() as f32;
                    let angle = (dy as f32).atan2(dx as f32);
                    let w = *width as f32;
                    let cx = (from.x + to.x) as f32 * 0.5;
                    let cy = (from.y + to.y) as f32 * 0.5;
                    self.rect_instances.push(InstanceData {
                        bounds: [cx - length * 0.5, cy - w * 0.5, length, w],
                        color: color_linear(*stroke),
                        params: [w * 0.5, angle, 0.0, 0.0],
                        border_color: [0.0; 4],
                        border_widths: [0.0; 4],
                    });
                }
                Primitive::Triangle { vertices, fill } => {
                    let [v0, v1, v2] = vertices;
                    let min_x = v0.x.min(v1.x).min(v2.x) as f32;
                    let min_y = v0.y.min(v1.y).min(v2.y) as f32;
                    let max_x = v0.x.max(v1.x).max(v2.x) as f32;
                    let max_y = v0.y.max(v1.y).max(v2.y) as f32;
                    let w = max_x - min_x;
                    let h = max_y - min_y;
                    if w < 0.5 || h < 0.5 {
                        continue;
                    }
                    // Store vertices in UV space [0,1]² relative to AABB
                    let uv = |p: &Point| ((p.x as f32 - min_x) / w, (p.y as f32 - min_y) / h);
                    let (u0x, u0y) = uv(v0);
                    let (u1x, u1y) = uv(v1);
                    let (u2x, u2y) = uv(v2);
                    self.rect_instances.push(InstanceData {
                        bounds: [min_x, min_y, w, h],
                        color: color_linear(*fill),
                        params: [0.0, 0.0, 1.0, 0.0], // params.z = 1.0 → triangle mode
                        border_color: [u0x, u0y, u1x, u1y],
                        border_widths: [u2x, u2y, 0.0, 0.0],
                    });
                }
                _ => {}
            }
        }
        // Flush remaining rects after the last clip change.
        flush_batch!();

        let n = self.rect_instances.len().min(MAX_INSTANCES);
        if n > 0 {
            self.queue
                .write_buffer(&self.ib, 0, bytemuck::cast_slice(&self.rect_instances[..n]));
        }

        // ── Text shaping (cached) with clip-aware TextBounds ────────────
        self.text_viewport.update(
            &self.queue,
            Resolution {
                width: vw,
                height: vh,
            },
        );

        let mut text_keys: Vec<(u64, f32, f32, glyphon::Color, TextBounds)> = Vec::new();
        let mut text_clip_stack: Vec<[u32; 4]> = Vec::new();
        let mut text_scissor: Option<[u32; 4]> = None;

        for p in &list.primitives {
            match p {
                Primitive::PushClip { bounds } => {
                    let new = [
                        bounds.origin.x.max(0.0) as u32,
                        bounds.origin.y.max(0.0) as u32,
                        (bounds.size.w().ceil() as u32).min(vw),
                        (bounds.size.h().ceil() as u32).min(vh),
                    ];
                    let clipped = match text_scissor {
                        Some(prev) => intersect_clip(prev, new),
                        None => new,
                    };
                    text_clip_stack.push(text_scissor.unwrap_or([0, 0, vw, vh]));
                    text_scissor = Some(clipped);
                }
                Primitive::PopClip => {
                    text_scissor = text_clip_stack.pop();
                    if text_scissor == Some([0, 0, vw, vh]) {
                        text_scissor = None;
                    }
                }
                Primitive::Text {
                    anchor,
                    content,
                    font_size,
                    color,
                } => {
                    let fs = *font_size as f32;
                    let key = text_cache_key(content, fs);
                    if !self.text_cache.contains_key(&key) {
                        let buf = shape_text_into(
                            &mut self.font_system,
                            content,
                            fs,
                            Some(vw as f32),
                            Some(vh as f32),
                        );
                        self.text_cache.insert(key, buf);
                    }
                    let gc = glyphon::Color::rgba(color.r, color.g, color.b, color.a);
                    let tb = match text_scissor {
                        Some([x, y, w, h]) => TextBounds {
                            left: x as i32,
                            top: y as i32,
                            right: (x + w) as i32,
                            bottom: (y + h) as i32,
                        },
                        None => TextBounds {
                            left: 0,
                            top: 0,
                            right: vw as i32,
                            bottom: vh as i32,
                        },
                    };
                    text_keys.push((key, anchor.x as f32, anchor.y as f32, gc, tb));
                }
                _ => {}
            }
        }

        let text_areas: Vec<TextArea> = text_keys
            .iter()
            .filter_map(|(key, x, y, color, tb)| {
                let buf = self.text_cache.get(key)?;
                Some(TextArea {
                    buffer: buf,
                    left: *x,
                    top: *y - buf.metrics().font_size * GLYPHON_TEXT_TOP_RATIO,
                    scale: 1.0,
                    bounds: *tb,
                    default_color: *color,
                    custom_glyphs: &[],
                })
            })
            .collect();

        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.text_viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();

        batches
    }

    fn draw<'a>(
        &'a self,
        enc: &'a mut wgpu::CommandEncoder,
        view: &'a wgpu::TextureView,
        batches: &[DrawBatch],
    ) {
        let (vw, vh) = (self.config.width, self.config.height);
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear({
                        let lc = color_linear(self.clear);
                        wgpu::Color {
                            r: lc[0] as f64,
                            g: lc[1] as f64,
                            b: lc[2] as f64,
                            a: 1.0,
                        }
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        if !batches.is_empty() {
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bg, &[]);
            rp.set_vertex_buffer(0, self.vb.slice(..));
            rp.set_vertex_buffer(1, self.ib.slice(..));
            for batch in batches {
                match batch.scissor {
                    Some([x, y, w, h]) if w > 0 && h > 0 => {
                        let cx = x.min(vw);
                        let cy = y.min(vh);
                        let cw = w.min(vw.saturating_sub(cx));
                        let ch = h.min(vh.saturating_sub(cy));
                        if cw == 0 || ch == 0 {
                            continue;
                        }
                        rp.set_scissor_rect(cx, cy, cw, ch);
                    }
                    Some(_) => continue, // zero-area clip — skip entirely
                    None => {
                        rp.set_scissor_rect(0, 0, vw, vh);
                    }
                }
                let end = (batch.start + batch.count).min(MAX_INSTANCES as u32);
                if batch.start < end {
                    rp.draw(0..4, batch.start..end);
                }
            }
        }
        // Reset scissor to full viewport before text rendering — the last
        // rect batch may have left a clip-scoped scissor active.
        rp.set_scissor_rect(0, 0, vw, vh);
        self.text_renderer
            .render(&self.text_atlas, &self.text_viewport, &mut rp)
            .unwrap();
    }

    // ── Public render paths ─────────────────────────────────────────────────

    /// Render to the window surface and present.  No-op in headless mode.
    pub fn paint(&mut self, list: &RenderList) {
        if self.surface.is_none() {
            return;
        }
        let batches = self.prepare(list);
        let surface = self.surface.as_ref().unwrap();
        let Ok(output) = surface.get_current_texture() else {
            return;
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.draw(&mut enc, &view, &batches);
        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        self.text_atlas.trim();
    }

    /// Render to an offscreen texture and read back as RGBA `Vec<u8>`.
    ///
    /// Returns `(width, height, rgba_pixels)`.
    pub fn capture(&mut self, list: &RenderList) -> (u32, u32, Vec<u8>) {
        let (w, h) = (self.config.width, self.config.height);
        let batches = self.prepare(list);

        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.draw(&mut enc, &view, &batches);

        let bpr = Self::aligned_bytes_per_row(w);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let row_bytes = (w * 4) as usize;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h as usize {
            let start = row * bpr as usize;
            rgba.extend_from_slice(&data[start..start + row_bytes]);
        }
        drop(data);
        readback.unmap();
        // BGRA → RGBA swap when the surface format stores blue first.
        if self.config.format == wgpu::TextureFormat::Bgra8UnormSrgb
            || self.config.format == wgpu::TextureFormat::Bgra8Unorm
        {
            for chunk in rgba.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
        }
        self.text_atlas.trim();
        (w, h, rgba)
    }

    /// WGPU requires `bytes_per_row` aligned to 256.
    fn aligned_bytes_per_row(width: u32) -> u32 {
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        (unpadded + align - 1) / align * align
    }

    /// Capture and save directly to a PNG file.
    pub fn capture_png(&mut self, list: &RenderList, path: &std::path::Path) {
        let (w, h, pixels) = self.capture(list);
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = png::Encoder::new(file, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&pixels).unwrap();
    }

    /// Measure the shaped width of `text` at `font_size` using the real font system.
    ///
    /// This gives accurate glyph-based widths that match the rendered output,
    /// unlike the `CHAR_WIDTH_RATIO` estimate in the DOM crate.
    pub fn measure_text(&mut self, text: &str, font_size: f64) -> f64 {
        let buf = shape_text_into(
            &mut self.font_system,
            text,
            font_size as f32,
            Some(MEASURE_MAX_WIDTH),
            None,
        );
        buf.layout_runs()
            .map(|run| run.line_w)
            .fold(0.0_f32, f32::max) as f64
    }

    /// Invalidate all cached text shapes (call when text content changes globally).
    pub fn invalidate_text_cache(&mut self) {
        self.text_cache.clear();
    }
}

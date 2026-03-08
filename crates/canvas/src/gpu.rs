//! Reusable WGPU renderer — instanced SDF rects + glyphon text.
//!
//! Supports both windowed (`init`) and headless (`init_headless`) modes.
//! The shader is loaded from `shaders/rect.wgsl` via `include_str!`.

use crate::theme;
use any_compute_core::render::{Color, Primitive, RenderList};
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache as GlyphCache, Family, FontSystem, Metrics, Resolution,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use pollster::block_on;
use std::sync::Arc;
use winit::window::Window;

/// SDF rounded-rect + border shader, loaded from file (no inline WGSL).
const SHADER_CODE: &str = include_str!("../shaders/rect.wgsl");

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
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniforms {
    screen_size: [f32; 2],
    _pad: [f32; 2],
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
    max_inst: usize,
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_viewport: Viewport,
    text_renderer: TextRenderer,
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
            present_mode: wgpu::PresentMode::AutoVsync,
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
        let max_inst = 50_000;
        let ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ib"),
            size: (std::mem::size_of::<InstanceData>() * max_inst) as u64,
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
            max_inst,
            font_system,
            swash_cache,
            text_atlas,
            text_viewport,
            text_renderer,
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

    fn prepare(&mut self, list: &RenderList) -> usize {
        let mut instances: Vec<InstanceData> = Vec::with_capacity(list.len());
        for p in &list.primitives {
            if let Primitive::Rect {
                bounds,
                fill,
                border,
                corner_radius,
            } = p
            {
                let (bw, bc) = border
                    .map(|b| (b.width as f32, color_linear(b.color)))
                    .unwrap_or((0.0, [0.0; 4]));
                instances.push(InstanceData {
                    bounds: [
                        bounds.origin.x as f32,
                        bounds.origin.y as f32,
                        bounds.size.w as f32,
                        bounds.size.h as f32,
                    ],
                    color: color_linear(*fill),
                    params: [*corner_radius as f32, bw, 0.0, 0.0],
                    border_color: bc,
                });
            }
        }
        let n = instances.len().min(self.max_inst);
        if n > 0 {
            self.queue
                .write_buffer(&self.ib, 0, bytemuck::cast_slice(&instances[..n]));
        }

        let (w, h) = (self.config.width, self.config.height);
        self.text_viewport
            .update(&self.queue, Resolution { width: w, height: h });

        let mut text_buffers: Vec<(GlyphBuffer, f32, f32, glyphon::Color)> = Vec::new();
        for p in &list.primitives {
            if let Primitive::Text {
                anchor,
                content,
                font_size,
                color,
            } = p
            {
                let sz = *font_size as f32;
                let mut buf = GlyphBuffer::new(&mut self.font_system, Metrics::new(sz, sz * 1.2));
                buf.set_size(&mut self.font_system, Some(w as f32), Some(h as f32));
                buf.set_text(
                    &mut self.font_system,
                    content,
                    Attrs::new().family(Family::SansSerif),
                    Shaping::Advanced,
                );
                buf.shape_until_scroll(&mut self.font_system, false);
                let gc = glyphon::Color::rgba(color.r, color.g, color.b, color.a);
                text_buffers.push((buf, anchor.x as f32, anchor.y as f32, gc));
            }
        }

        let text_areas: Vec<TextArea> = text_buffers
            .iter()
            .map(|(buf, x, y, color)| TextArea {
                buffer: buf,
                left: *x,
                top: *y - buf.metrics().font_size * 0.8,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: w as i32,
                    bottom: h as i32,
                },
                default_color: *color,
                custom_glyphs: &[],
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

        n
    }

    fn draw<'a>(
        &'a self,
        enc: &'a mut wgpu::CommandEncoder,
        view: &'a wgpu::TextureView,
        n: usize,
    ) {
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
        if n > 0 {
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bg, &[]);
            rp.set_vertex_buffer(0, self.vb.slice(..));
            rp.set_vertex_buffer(1, self.ib.slice(..));
            rp.draw(0..4, 0..n as u32);
        }
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
        let n = self.prepare(list);
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
        self.draw(&mut enc, &view, n);
        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        self.text_atlas.trim();
    }

    /// Render to an offscreen texture and read back as RGBA `Vec<u8>`.
    ///
    /// Returns `(width, height, rgba_pixels)`.
    pub fn capture(&mut self, list: &RenderList) -> (u32, u32, Vec<u8>) {
        let (w, h) = (self.config.width, self.config.height);
        let n = self.prepare(list);

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
        self.draw(&mut enc, &view, n);

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
        let (w, h, mut pixels) = self.capture(list);
        if self.config.format == wgpu::TextureFormat::Bgra8UnormSrgb
            || self.config.format == wgpu::TextureFormat::Bgra8Unorm
        {
            for chunk in pixels.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
        }
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = png::Encoder::new(file, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&pixels).unwrap();
    }
}

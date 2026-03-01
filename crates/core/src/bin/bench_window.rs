//! Native WGPU benchmark dashboard — zero frameworks, zero DOM, zero overhead.
//!
//! Renders directly to GPU via instanced draw calls.
//! Runs real `any_compute_core::bench` workloads on background threads and
//! streams results into the render loop at 60+ FPS with zero stutter.

use any_compute_core::bench::*;
use any_compute_core::dom::{style::*, tree::*};
use any_compute_core::kernel::{UnaryOp, best_kernel};
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::{Color, Primitive, RenderList};
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache as GlyphCache, Family, FontSystem, Metrics, Resolution,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use pollster::block_on;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// ── Theme ────────────────────────────────────────────────────────────────────
mod theme {
    use super::Color;
    pub const BG: Color = Color::rgb(30, 30, 46);
    pub const SURFACE: Color = Color::rgb(49, 50, 68);
    pub const SURFACE_BRIGHT: Color = Color::rgb(69, 71, 90);
    pub const GREEN: Color = Color::rgb(166, 227, 161);
    pub const BLUE: Color = Color::rgb(137, 180, 250);
    pub const RED: Color = Color::rgb(243, 139, 168);
    pub const YELLOW: Color = Color::rgb(249, 226, 175);
    pub const MAUVE: Color = Color::rgb(203, 166, 247);
    pub const TEXT: Color = Color::rgb(205, 214, 244);
    pub const TEXT_DIM: Color = Color::rgb(147, 153, 178);
    pub const SIDEBAR_BG: Color = Color::rgb(24, 24, 37);
    pub const ACCENT: Color = Color::rgb(137, 180, 250);
}

// ── WGPU Renderer (reusable) ────────────────────────────────────────────────
const SHADER_CODE: &str = r#"
struct VertexInput { @location(0) position: vec2<f32> };
struct InstanceInput {
    @location(1) bounds: vec4<f32>,
    @location(2) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
struct Uniforms { screen_size: vec2<f32> }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = vec2<f32>(
        instance.bounds.x + model.position.x * instance.bounds.z,
        instance.bounds.y + model.position.y * instance.bounds.w
    );
    let clip_x = (pos.x / uniforms.screen_size.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (pos.y / uniforms.screen_size.y) * 2.0;
    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> { return in.color; }
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    bounds: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniforms {
    screen_size: [f32; 2],
    _pad: [f32; 2],
}

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    ub: wgpu::Buffer,
    bg: wgpu::BindGroup,
    max_inst: usize,
    // Text rendering (glyphon)
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_viewport: Viewport,
    text_renderer: TextRenderer,
}

impl Gpu {
    fn init(window: Arc<winit::window::Window>) -> Self {
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
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
            screen_size: [size.width as f32, size.height as f32],
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
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // ── Glyphon text rendering ──
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
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        let u = GpuUniforms {
            screen_size: [w as f32, h as f32],
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.ub, 0, bytemuck::bytes_of(&u));
    }

    fn paint(&mut self, list: &RenderList) {
        // ── Collect rect instances ──
        let mut instances: Vec<InstanceData> = Vec::with_capacity(list.len());
        for p in &list.primitives {
            if let Primitive::Rect { bounds, fill, .. } = p {
                instances.push(InstanceData {
                    bounds: [
                        bounds.origin.x as f32,
                        bounds.origin.y as f32,
                        bounds.size.w as f32,
                        bounds.size.h as f32,
                    ],
                    color: [
                        fill.r as f32 / 255.0,
                        fill.g as f32 / 255.0,
                        fill.b as f32 / 255.0,
                        fill.a as f32 / 255.0,
                    ],
                });
            }
        }
        let n = instances.len().min(self.max_inst);
        if n > 0 {
            self.queue
                .write_buffer(&self.ib, 0, bytemuck::cast_slice(&instances[..n]));
        }

        // ── Collect text primitives → glyphon TextAreas ──
        let w = self.config.width;
        let h = self.config.height;
        self.text_viewport.update(
            &self.queue,
            Resolution {
                width: w,
                height: h,
            },
        );

        // Build one glyphon Buffer per Text primitive
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
                top: *y - buf.metrics().font_size * 0.8, // baseline offset
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

        // ── Render pass ──
        let Ok(output) = self.surface.get_current_texture() else {
            return;
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: theme::BG.r as f64 / 255.0,
                            g: theme::BG.g as f64 / 255.0,
                            b: theme::BG.b as f64 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // Draw rects
            if n > 0 {
                rp.set_pipeline(&self.pipeline);
                rp.set_bind_group(0, &self.bg, &[]);
                rp.set_vertex_buffer(0, self.vb.slice(..));
                rp.set_vertex_buffer(1, self.ib.slice(..));
                rp.draw(0..4, 0..n as u32);
            }
            // Draw text on top
            self.text_renderer
                .render(&self.text_atlas, &self.text_viewport, &mut rp)
                .unwrap();
        }
        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        self.text_atlas.trim();
    }
}

// ── Shared async state ──────────────────────────────────────────────────────
#[derive(Clone)]
struct SharedState {
    inner: Arc<Mutex<AppData>>,
}

struct AppData {
    hw: Option<HardwareReport>,
    bench_results: Vec<ScenarioReport>,
    bench_running: bool,
    bench_progress: (usize, usize),
    current_cat: Option<String>,
    // Live simulation
    sim_running: bool,
    ac_ops: f64,
    rayon_ops: f64,
    std_ops: f64,
    // Tab
    tab: usize,
    scroll_y: f64,
}

impl SharedState {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppData {
                hw: None,
                bench_results: Vec::new(),
                bench_running: false,
                bench_progress: (0, 0),
                current_cat: None,
                sim_running: false,
                ac_ops: 0.0,
                rayon_ops: 0.0,
                std_ops: 0.0,
                tab: 0,
                scroll_y: 0.0,
            })),
        }
    }

    fn read<R>(&self, f: impl FnOnce(&AppData) -> R) -> R {
        f(&self.inner.lock().unwrap())
    }

    fn write<R>(&self, f: impl FnOnce(&mut AppData) -> R) -> R {
        f(&mut self.inner.lock().unwrap())
    }
}

// ── Background workers ──────────────────────────────────────────────────────
fn spawn_hw_detect(state: SharedState) {
    std::thread::spawn(move || {
        let hw = detect_hardware();
        state.write(|d| d.hw = Some(hw));
    });
}

fn spawn_benchmarks(state: SharedState) {
    if state.read(|d| d.bench_running) {
        return;
    }
    state.write(|d| {
        d.bench_running = true;
        d.bench_results.clear();
        d.bench_progress = (0, BenchCategory::ALL.len());
    });

    std::thread::spawn(move || {
        for (i, &cat) in BenchCategory::ALL.iter().enumerate() {
            state.write(|d| d.current_cat = Some(cat.label().to_string()));
            match std::panic::catch_unwind(|| run_category(cat)) {
                Ok(report) => {
                    state.write(|d| {
                        d.bench_results.push(report);
                        d.bench_progress.0 = i + 1;
                    });
                }
                Err(_) => {
                    state.write(|d| {
                        let mut r = ScenarioReport::default();
                        r.category = format!("{} (CRASHED)", cat.label());
                        d.bench_results.push(r);
                        d.bench_progress.0 = i + 1;
                    });
                }
            }
        }
        state.write(|d| {
            d.bench_running = false;
            d.current_cat = None;
        });
    });
}

fn spawn_simulation(state: SharedState) {
    if state.read(|d| d.sim_running) {
        state.write(|d| d.sim_running = false);
        return;
    }
    state.write(|d| {
        d.sim_running = true;
        d.ac_ops = 0.0;
        d.rayon_ops = 0.0;
        d.std_ops = 0.0;
    });

    fn throughput_loop(
        s: SharedState,
        compute: impl Fn(&[f64]) -> Vec<f64> + Send + 'static,
        report: impl Fn(&mut AppData, f64) + Send + 'static,
    ) {
        std::thread::spawn(move || {
            let data = vec![1.0_f64; 200_000];
            let mut last = Instant::now();
            let mut ops = 0usize;
            while s.read(|d| d.sim_running) {
                std::hint::black_box(&compute(&data));
                ops += data.len();
                let el = last.elapsed().as_secs_f64();
                if el > 0.3 {
                    s.write(|d| report(d, ops as f64 / el));
                    ops = 0;
                    last = Instant::now();
                }
            }
        });
    }

    throughput_loop(
        state.clone(),
        |data| {
            let kern = best_kernel();
            kern.map_unary_f64(data, UnaryOp::Sigmoid)
        },
        |d, t| d.ac_ops = t,
    );

    throughput_loop(
        state.clone(),
        |data| data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        |d, t| d.rayon_ops = t,
    );

    throughput_loop(
        state.clone(),
        |data| data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        |d, t| d.std_ops = t,
    );
}

// ── Layout / Paint ──────────────────────────────────────────────────────────
const SIDEBAR_W: f64 = 220.0;
const HEADER_H: f64 = 56.0;
const TAB_LABELS: &[&str] = &["Hardware", "Benchmarks", "Live Showdown"];

// ── Semantic style presets ──────────────────────────────────────────────────
fn s_title() -> Style {
    Style::default().font(22.0).color(theme::TEXT)
}
fn s_subtitle() -> Style {
    Style::default().font(12.0).color(theme::TEXT_DIM)
}
fn s_label() -> Style {
    Style::default().font(11.0).color(theme::TEXT_DIM)
}
fn s_body() -> Style {
    Style::default().font(12.0).color(theme::TEXT)
}
fn s_card() -> Style {
    Style::default()
        .grow(1.0)
        .bg(theme::SURFACE)
        .radius(12.0)
        .pad(16.0)
        .gap(6.0)
}

// ── DOM construction helpers ────────────────────────────────────────────────
/// Section header row with title + flex spacer. Returns the row NodeId for button attachment.
fn section_hdr(t: &mut Tree, parent: NodeId, title: &str) -> NodeId {
    let hdr = t.add_box(parent, Style::default().row().align(Align::Center));
    t.add_text(hdr, title, s_title());
    t.add_box(hdr, Style::default().grow(1.0));
    hdr
}

/// Action button: colored pill with label + tag.
fn action_btn(
    t: &mut Tree,
    parent: NodeId,
    label: &str,
    bg: Color,
    fg: Color,
    tag: &str,
) -> NodeId {
    let btn = t.add_box(
        parent,
        Style::default()
            .bg(bg)
            .radius(8.0)
            .pad_xy(16.0, 8.0)
            .row()
            .align(Align::Center),
    );
    t.add_text(btn, label, Style::default().font(12.0).color(fg));
    t.tag(btn, tag);
    btn
}

/// Card with title accent. Returns card NodeId.
fn card(t: &mut Tree, parent: NodeId, title: &str, accent: Color) -> NodeId {
    let c = t.add_box(parent, s_card());
    t.add_text(c, title, Style::default().font(16.0).color(accent));
    c
}

/// Build the full DOM tree from current app state.
fn build_tree(state: &SharedState, w: f64, h: f64) -> Tree {
    let data = state.inner.lock().unwrap();
    let mut t = Tree::new(Style::default().w(w).h(h).row().bg(theme::BG));
    let root = t.root;

    // ── Sidebar ──
    let sb = t.add_box(
        root,
        Style::default()
            .w(SIDEBAR_W)
            .bg(theme::SIDEBAR_BG)
            .pad_xy(12.0, 16.0)
            .gap(8.0),
    );
    let brand = t.add_box(
        sb,
        Style::default()
            .row()
            .gap(12.0)
            .h(40.0)
            .align(Align::Center),
    );
    t.add_box(
        brand,
        Style::default()
            .w(28.0)
            .h(28.0)
            .bg(theme::ACCENT)
            .radius(6.0),
    );
    let btext = t.add_box(brand, Style::default().gap(2.0));
    t.add_text(
        btext,
        "any-compute",
        Style::default().font(16.0).color(theme::TEXT),
    );
    t.add_text(
        btext,
        "v0.5.0",
        Style::default().font(10.0).color(theme::TEXT_DIM),
    );
    t.add_box(sb, Style::default().h(12.0));
    for (i, &label) in TAB_LABELS.iter().enumerate() {
        let active = data.tab == i;
        let (bg, fg) = if active {
            (theme::ACCENT, theme::SIDEBAR_BG)
        } else {
            (Color::TRANSPARENT, theme::TEXT_DIM)
        };
        let btn = t.add_box(
            sb,
            Style::default()
                .h(36.0)
                .bg(bg)
                .radius(8.0)
                .pad_xy(12.0, 0.0)
                .row()
                .align(Align::Center),
        );
        t.add_text(btn, label, Style::default().font(13.0).color(fg));
        t.tag(btn, format!("tab-{}", i));
    }

    // ── Main area ──
    let main_col = t.add_box(root, Style::default().grow(1.0));
    let hdr = t.add_box(
        main_col,
        Style::default()
            .h(HEADER_H)
            .bg(theme::SURFACE)
            .row()
            .pad_xy(24.0, 0.0)
            .align(Align::Center),
    );
    t.add_text(
        hdr,
        TAB_LABELS[data.tab],
        Style::default().font(18.0).color(theme::TEXT),
    );
    let content = t.add_box(
        main_col,
        Style::default()
            .grow(1.0)
            .overflow(Overflow::Scroll)
            .pad(24.0)
            .gap(16.0),
    );
    t.arena[content.0].scroll.y = data.scroll_y;

    match data.tab {
        0 => build_hw(&mut t, content, &data),
        1 => build_bench(&mut t, content, &data),
        2 => build_sim(&mut t, content, &data),
        _ => {}
    }

    t
}

fn build_hw(t: &mut Tree, p: NodeId, data: &AppData) {
    t.add_text(p, "Hardware Profile", s_title());
    t.add_text(p, "Detected system capabilities", s_subtitle());

    let Some(hw) = &data.hw else {
        t.add_text(
            p,
            "Detecting hardware\u{2026}",
            Style::default().font(14.0).color(theme::YELLOW),
        );
        return;
    };

    let row = t.add_box(p, Style::default().row().gap(12.0));

    // ── CPU ──
    let c = card(t, row, "Processor", theme::ACCENT);
    let topo = format!(
        "{} cores / {} threads",
        hw.cpu.physical_cores, hw.cpu.logical_cores
    );
    let freq = format_hz(hw.cpu.frequency_mhz);
    for (lbl, val) in [
        ("Brand", hw.cpu.brand.as_str()),
        ("Arch", hw.cpu.arch.as_str()),
        ("Topology", topo.as_str()),
        ("Frequency", freq.as_str()),
    ] {
        let r = t.add_box(c, Style::default().row().gap(8.0));
        t.add_text(r, lbl, s_label().w(72.0));
        t.add_text(r, val, s_body());
    }

    // ── SIMD ──
    let c = card(t, row, "SIMD / Vector", theme::GREEN);
    t.add_text(c, &hw.simd.detected, s_body());
    t.add_text(
        c,
        &format!("{}-bit vectors", hw.simd.vector_width),
        s_label(),
    );
    let tags_str = hw.simd.features.join("  \u{00b7}  ");
    t.add_text(
        c,
        &tags_str,
        Style::default().font(10.0).color(theme::MAUVE),
    );

    // ── Memory & GPU ──
    let c = card(t, row, "Memory & GPU", theme::YELLOW);
    let total_gb = hw.memory.total_bytes / 1024 / 1024 / 1024;
    let avail_gb = hw.memory.available_bytes / 1024 / 1024 / 1024;
    t.add_text(
        c,
        &format!("{} GB total / {} GB available", total_gb, avail_gb),
        s_body(),
    );
    let pct = if hw.memory.total_bytes > 0 {
        hw.memory.used_bytes as f64 / hw.memory.total_bytes as f64
    } else {
        0.0
    };
    let bar_c = if pct > 0.8 { theme::RED } else { theme::GREEN };
    t.add_bar(c, pct, bar_c, Style::default().h(8.0).radius(4.0));
    t.add_text(
        c,
        &format!("{:.0}% used", pct * 100.0),
        Style::default().font(10.0).color(theme::TEXT_DIM),
    );
    for gpu in &hw.gpus {
        let g = t.add_box(
            c,
            Style::default()
                .bg(theme::SURFACE_BRIGHT)
                .radius(6.0)
                .pad_xy(8.0, 6.0),
        );
        t.add_text(g, &gpu.name, s_label());
    }
    if hw.gpus.is_empty() {
        t.add_text(c, "No GPU detected", s_label());
    }
}

fn build_bench(t: &mut Tree, p: NodeId, data: &AppData) {
    let hdr = section_hdr(t, p, "Benchmark Results");
    let (btn_bg, btn_fg, btn_lbl) = if data.bench_running {
        (theme::SURFACE_BRIGHT, theme::TEXT_DIM, "Running\u{2026}")
    } else {
        (theme::GREEN, theme::SIDEBAR_BG, "Run All Tests")
    };
    if !data.bench_running {
        action_btn(t, hdr, btn_lbl, btn_bg, btn_fg, "run-bench");
    } else {
        let btn = t.add_box(
            hdr,
            Style::default()
                .bg(btn_bg)
                .radius(8.0)
                .pad_xy(16.0, 8.0)
                .row()
                .align(Align::Center),
        );
        t.add_text(btn, btn_lbl, Style::default().font(12.0).color(btn_fg));
    }

    if data.bench_running {
        let (done, total) = data.bench_progress;
        let pct = done as f64 / total.max(1) as f64;
        t.add_bar(p, pct, theme::ACCENT, Style::default().h(6.0).radius(3.0));
        if let Some(cat) = &data.current_cat {
            t.add_text(
                p,
                &format!("Running: {} ({}/{})", cat, done, total),
                s_label(),
            );
        }
    }

    if data.bench_results.is_empty() && !data.bench_running {
        t.add_text(
            p,
            "No results yet \u{2014} click 'Run All Tests' to begin.",
            Style::default().font(14.0).color(theme::TEXT_DIM),
        );
        return;
    }

    let mut i = 0;
    while i < data.bench_results.len() {
        let row = t.add_box(p, Style::default().row().gap(12.0));
        for j in 0..2 {
            let idx = i + j;
            if idx >= data.bench_results.len() {
                break;
            }
            let report = &data.bench_results[idx];
            let c = t.add_box(
                row,
                Style::default()
                    .grow(1.0)
                    .bg(theme::SURFACE)
                    .radius(10.0)
                    .pad(14.0)
                    .gap(4.0),
            );
            t.add_text(
                c,
                &report.category,
                Style::default().font(14.0).color(theme::ACCENT),
            );
            let max_ops = report
                .results
                .iter()
                .map(|r| r.throughput_ops_sec)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            for (bi, r) in report.results.iter().take(6).enumerate() {
                let name = if r.name.len() > 20 {
                    format!("{}\u{2026}", &r.name[..18])
                } else {
                    r.name.clone()
                };
                let pct = r.throughput_ops_sec / max_ops;
                let bar_c = [theme::GREEN, theme::BLUE, theme::YELLOW, theme::MAUVE][bi % 4];
                let entry = t.add_box(c, Style::default().gap(2.0));
                let lr = t.add_box(entry, Style::default().row());
                t.add_text(lr, &name, Style::default().font(9.0).color(theme::TEXT_DIM));
                t.add_box(lr, Style::default().grow(1.0));
                t.add_text(
                    lr,
                    &format_ops(r.throughput_ops_sec),
                    Style::default().font(9.0).color(theme::TEXT),
                );
                t.add_bar(entry, pct, bar_c, Style::default().h(9.0).radius(3.0));
            }
        }
        i += 2;
    }
}

fn build_sim(t: &mut Tree, p: NodeId, data: &AppData) {
    let hdr = section_hdr(t, p, "Live Showdown");
    let (btn_bg, btn_lbl) = if data.sim_running {
        (theme::RED, "Stop Showdown")
    } else {
        (theme::GREEN, "Start Showdown")
    };
    action_btn(t, hdr, btn_lbl, btn_bg, theme::SIDEBAR_BG, "toggle-sim");

    t.add_text(
        p,
        "Real-time Sigmoid(200K): any-compute vs rayon vs stdlib",
        s_subtitle(),
    );

    let peak = data.ac_ops.max(data.rayon_ops).max(data.std_ops).max(1.0);
    for &(label, ops, color) in &[
        ("any-compute (vectorized kernel)", data.ac_ops, theme::GREEN),
        ("rayon (parallel iterator)", data.rayon_ops, theme::BLUE),
        (
            "stdlib (single-thread iterator)",
            data.std_ops,
            theme::YELLOW,
        ),
    ] {
        let lane = t.add_box(p, Style::default().gap(4.0));
        let top = t.add_box(lane, Style::default().row());
        t.add_text(top, label, Style::default().font(13.0).color(color));
        t.add_box(top, Style::default().grow(1.0));
        t.add_text(
            top,
            &format_ops(ops),
            Style::default().font(13.0).color(theme::TEXT),
        );
        let frac = if peak > 0.0 { ops / peak } else { 0.0 };
        t.add_bar(lane, frac, color, Style::default().h(24.0).radius(6.0));
        if ops != data.std_ops && data.std_ops > 0.0 {
            t.add_text(
                lane,
                &format!("{:.1}x vs stdlib", ops / data.std_ops),
                Style::default().font(10.0).color(theme::TEXT_DIM),
            );
        }
    }
}

// ── Click dispatch ──────────────────────────────────────────────────────────
fn handle_tag(state: &SharedState, tag: &str) {
    match tag {
        "tab-0" => state.write(|d| d.tab = 0),
        "tab-1" => state.write(|d| d.tab = 1),
        "tab-2" => state.write(|d| d.tab = 2),
        "run-bench" => spawn_benchmarks(state.clone()),
        "toggle-sim" => spawn_simulation(state.clone()),
        _ => {}
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────
pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute \u{2014} Benchmark Dashboard")
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 900.0))
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());
    let state = SharedState::new();
    spawn_hw_detect(state.clone());

    let mut fps_timer = Instant::now();
    let mut fps_count = 0u32;
    let mut cursor_pos = (0.0_f64, 0.0_f64);
    let mut last_tree: Option<Tree> = None;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            WindowEvent::CloseRequested => {
                state.write(|d| d.sim_running = false);
                elwt.exit();
            }
            WindowEvent::Resized(s) => gpu.resize(s.width, s.height),
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if let Some(tree) = &last_tree {
                    let pos = Point::new(cursor_pos.0, cursor_pos.1);
                    if let Some(tag) = tree.click(pos) {
                        handle_tag(&state, tag);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                cursor_pos = (position.x, position.y);
            }
            WindowEvent::RedrawRequested => {
                let sz = window.inner_size();
                if sz.width > 0 && sz.height > 0 {
                    let w = sz.width as f64;
                    let h = sz.height as f64;
                    let mut tree = build_tree(&state, w, h);
                    tree.layout(Size::new(w, h));
                    let mut list = RenderList::default();
                    tree.paint(&mut list);
                    gpu.paint(&list);
                    fps_count += 1;
                    if fps_timer.elapsed().as_secs() >= 1 {
                        window.set_title(&format!(
                            "any-compute \u{2014} {} FPS | {} nodes | {} primitives",
                            fps_count,
                            tree.arena.len(),
                            list.len(),
                        ));
                        fps_count = 0;
                        fps_timer = Instant::now();
                    }
                    last_tree = Some(tree);
                }
                window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y,
                };
                state.write(|d| d.scroll_y = (d.scroll_y - dy).max(0.0));
            }
            _ => {}
        },
        _ => {}
    });
}

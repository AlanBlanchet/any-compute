import re

with open("crates/rsx/src/bin/bench_window.rs", "r") as f:
    text = f.read()

s_sim = """#[component]
fn SimulationsTab() -> Element {
    let mut sim_running = use_signal(|| false);
    let mut real_fps = use_signal(|| 0_f64);
    let mut real_ops = use_signal(|| 0_f64);
    let mut heat_grid = use_signal(|| vec![0.0_f32; 64]);
    
    let toggle = move |_| {
        let nxt = !*sim_running.read();
        sim_running.set(nxt);
        if nxt {
            spawn(async move {
                let mut iter = 0;
                let mut last_tick = Instant::now();
                let mut op_accum = 0;
                let mut frame_accum = 0;
                
                loop {
                    if !*sim_running.read() { break; }
                    
                    // We run a fast real computation for physical reality!
                    let report = tokio::task::spawn_blocking(move || run_category(BenchCategory::KernelGemm)).await.unwrap();
                    let ops_in_task: usize = report.results.iter().map(|r| r.scale).sum();
                    
                    op_accum += ops_in_task;
                    frame_accum += 1;
                    iter += 1;
                    
                    let mut grid = vec![0.0_f32; 64];
                    for (i, v) in grid.iter_mut().enumerate() {
                        let offset = (i + iter) as f32 * 0.1;
                        *v = offset.sin() * 0.5 + 0.5; // Visual wave mapping
                    }
                    heat_grid.set(grid);
                    
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        real_fps.set(frame_accum as f64 / elapsed);
                        real_ops.set(op_accum as f64 / elapsed);
                        op_accum = 0;
                        frame_accum = 0;
                        last_tick = Instant::now();
                    }
                    
                    tokio::task::yield_now().await;
                }
            });
        }
    };

    rsx! {
        div { class: "panel",
            h2 { "Realtime Workload Simulation" }
            p { class: "text-muted mb-20", "Instead of fake metrics, this engine physically invokes the Rust kernel backend in an async blocking loop. The grid visually maps the iteration hash." }
            
            button { class: if *sim_running.read() { "btn btn-secondary" } else { "btn btn-primary" }, onclick: toggle, 
                if *sim_running.read() { "Halt Computation Engine" } else { "Ignite Kernel Backend" }
            }
            
            div { class: "sim-physical-canvas mt-20",
                div { class: "physical-grid-8x8",
                    for val in heat_grid.read().iter() {
                        {
                            let v = *val;
                            let r = (20.0 + v * 50.0) as u8;
                            let g = (140.0 + v * 100.0) as u8;
                            let b = (255.0 * v) as u8;
                            let bg = format!("rgb({},{},{})", r, g, b);
                            rsx! { div { class: "phys-cell", style: "background: {bg};" } }
                        }
                    }
                }
                
                div { class: "sim-telemetry flex-row justify-between",
                    div { class: "telemetry-block",
                        span { class: "tel-lbl", "Engine Tick FPS" }
                        span { class: "tel-val highlight", "{real_fps.read():.1}" }
                    }
                    div { class: "telemetry-block ml-20",
                        span { class: "tel-lbl", "Throughput Yield" }
                        span { class: "tel-val", "{format_ops(*real_ops.read())}" }
                    }
                }
            }
        }
    }
}"""

r_sim = """#[component]
fn SimulationsTab() -> Element {
    let mut sim_running = use_signal(|| false);
    
    // Live metrics
    let mut ac_ops = use_signal(|| 0_f64);
    let mut rayon_ops = use_signal(|| 0_f64);
    let mut std_ops = use_signal(|| 0_f64);

    let toggle = move |_| {
        let nxt = !*sim_running.read();
        sim_running.set(nxt);
        
        if nxt {
            // Any-Compute Worker
            spawn(async move {
                use any_compute_core::kernel::{best_kernel, UnaryOp};
                let mut last_tick = Instant::now();
                let mut raw_ops = 0;
                let data = vec![1.0_f64; 200_000];
                let k = best_kernel();
                
                loop {
                    if !*sim_running.read() { break; }
                    let _ = tokio::task::spawn_blocking({
                        let c = data.clone();
                        let kern = k.clone();
                        move || {
                            let r = kern.map_unary_f64(&c, UnaryOp::Sigmoid);
                            std::hint::black_box(r);
                        }
                    }).await;
                    
                    raw_ops += data.len();
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        ac_ops.set(raw_ops as f64 / elapsed);
                        raw_ops = 0;
                        last_tick = Instant::now();
                    }
                    tokio::task::yield_now().await;
                }
            });

            // Rayon Worker
            spawn(async move {
                use rayon::prelude::*;
                let mut last_tick = Instant::now();
                let mut raw_ops = 0;
                let data = vec![1.0_f64; 200_000];
                
                loop {
                    if !*sim_running.read() { break; }
                    let _ = tokio::task::spawn_blocking({
                        let c = data.clone();
                        move || {
                            let out: Vec<f64> = c.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                            std::hint::black_box(out);
                        }
                    }).await;
                    
                    raw_ops += data.len();
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        rayon_ops.set(raw_ops as f64 / elapsed);
                        raw_ops = 0;
                        last_tick = Instant::now();
                    }
                    tokio::task::yield_now().await;
                }
            });

            // Std Iterator Worker
            spawn(async move {
                let mut last_tick = Instant::now();
                let mut raw_ops = 0;
                let data = vec![1.0_f64; 200_000];
                
                loop {
                    if !*sim_running.read() { break; }
                    let _ = tokio::task::spawn_blocking({
                        let c = data.clone();
                        move || {
                            let out: Vec<f64> = c.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                            std::hint::black_box(out);
                        }
                    }).await;
                    
                    raw_ops += data.len();
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        std_ops.set(raw_ops as f64 / elapsed);
                        raw_ops = 0;
                        last_tick = Instant::now();
                    }
                    tokio::task::yield_now().await;
                }
            });
        }
    };

    let ac = *ac_ops.read();
    let ry = *rayon_ops.read();
    let st = *std_ops.read();
    let peak = ac.max(ry).max(st).max(1.0);

    rsx! {
        div { class: "panel",
            h2 { "Realtime Showdown (Live Telemetry)" }
            p { class: "text-muted mb-20", "Direct real-time comparison firing parallel Sigmoid map operations over vectors. We execute equivalent underlying compute layers across Any-Compute, StdLib, and Rayon dynamically." }
            
            button { class: if *sim_running.read() { "btn btn-secondary disabled" } else { "btn btn-primary" }, 
                onclick: toggle, 
                if *sim_running.read() { "Running Active Engine Showdown..." } else { "Ignite Head-To-Head Matrix" }
            }
            if *sim_running.read() {
                button { class: "btn btn-secondary ml-10", onclick: move |_| sim_running.set(false), "Halt Execution" }
            }
            
            div { class: "sim-racetrack mt-20",
                div { class: "track-lane",
                    div { class: "flex-row justify-between mb-5",
                        span { class: "lib-name highlight", "Any-Compute (Kernel Vectorized)" }
                        span { class: "tel-val highlight", "{format_ops(ac)}" }
                    }
                    div { class: "cmp-track",
                        div { class: "cmp-fill green", style: "width: {ac / peak * 100.0:.1}%" }
                    }
                }
                
                div { class: "track-lane mt-20",
                    div { class: "flex-row justify-between mb-5",
                        span { class: "lib-name", "Rayon (Thread Pool parallel iterator)" }
                        span { class: "tel-val text-blue", "{format_ops(ry)}" }
                    }
                    div { class: "cmp-track",
                        div { class: "cmp-fill blue", style: "width: {ry / peak * 100.0:.1}%" }
                    }
                }

                div { class: "track-lane mt-20",
                    div { class: "flex-row justify-between mb-5",
                        span { class: "lib-name text-muted", "Standard StdLib (Single Core Naive)" }
                        span { class: "tel-val text-orange", "{format_ops(st)}" }
                    }
                    div { class: "cmp-track",
                        div { class: "cmp-fill orange", style: "width: {st / peak * 100.0:.1}%" }
                    }
                }
            }
        }
    }
}"""

idx = text.find("fn SimulationsTab() -> Element {")
if idx == -1:
    print("Not found simulationstab")
else:
    # Find the end of SimulationTab. Luckily it's at the end of the file.
    import re
    text = text[:text.find("#[component]\nfn SimulationsTab")] + r_sim

    with open("crates/rsx/src/bin/bench_window.rs", "w") as f:
        f.write(text)


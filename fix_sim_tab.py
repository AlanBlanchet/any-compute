import re

with open("crates/rsx/src/bin/bench_window.rs", "r") as f:
    text = f.read()

text = re.sub(
    r'let k = best_kernel\(\);\s*loop\s*\{\s*if\s*!\*sim_running\.read\(\)\s*\{\s*break;\s*\}\s*let _ = tokio::task::spawn_blocking\(\{\s*let c = data\.clone\(\);\s*let kern = k\.clone\(\);\s*move\s*\|\|\s*\{\s*let r = kern\.map_unary_f64\(&c, UnaryOp::Sigmoid\);',
    r'loop {\n                    if !*sim_running.read() { break; }\n                    let _ = tokio::task::spawn_blocking({\n                        let c = data.clone();\n                        move || {\n                            let kern = best_kernel();\n                            let r = kern.map_unary_f64(&c, UnaryOp::Sigmoid);',
    text,
    flags=re.MULTILINE | re.DOTALL
)

with open("crates/rsx/src/bin/bench_window.rs", "w") as f:
    f.write(text)

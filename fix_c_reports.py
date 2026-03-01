with open("crates/rsx/src/bin/bench_window.rs", "r") as f:
    text = f.read()

text = text.replace(
    "let cmp = tokio::task::spawn_blocking(move || build_comparison_tables(&c_reports, &[], &[])).await.unwrap();",
    "let cmp = tokio::task::spawn_blocking(move || build_comparison_tables(&c_reports, &c_reports, &c_reports)).await.unwrap();"
)

with open("crates/rsx/src/bin/bench_window.rs", "w") as f:
    f.write(text)

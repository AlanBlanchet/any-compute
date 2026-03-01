//! Generate cross-language bindings and tests.
//!
//! Usage: `cargo run --bin anc-codegen [--output-dir <dir>]`

use any_compute_ffi::codegen::{FfiRegistry, generate_all};
use std::path::PathBuf;

fn main() {
    let out_dir = std::env::args()
        .position(|a| a == "--output-dir")
        .and_then(|i| std::env::args().nth(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("bindings"));

    let registry = FfiRegistry::default_any_compute();

    println!(
        "Generating bindings for {} FFI functions → {}",
        registry.functions.len(),
        out_dir.display()
    );

    generate_all(&registry, &out_dir).expect("Failed to generate bindings");

    for (target, files) in [
        (
            "python",
            vec!["python/any_compute.py", "python/test_any_compute.py"],
        ),
        (
            "javascript",
            vec![
                "javascript/any_compute.js",
                "javascript/any_compute.d.ts",
                "javascript/any_compute.test.js",
            ],
        ),
        (
            "java",
            vec![
                "java/com/anycompute/AnyCompute.java",
                "java/com/anycompute/AnyComputeTest.java",
            ],
        ),
        (
            "react",
            vec![
                "react/src/hooks.ts",
                "react/src/bench.ts",
                "react/package.json",
            ],
        ),
        ("vue", vec!["vue/src/composables.ts", "vue/package.json"]),
        (
            "svelte",
            vec!["svelte/src/stores.ts", "svelte/package.json"],
        ),
        (
            "angular",
            vec![
                "angular/src/any-compute.service.ts",
                "angular/src/any-compute.module.ts",
                "angular/package.json",
            ],
        ),
        (
            "node",
            vec![
                "node/src/index.ts",
                "node/src/bench.ts",
                "node/package.json",
            ],
        ),
    ] {
        println!("  [{target}]");
        for f in files {
            println!("    {}/{f}", out_dir.display());
        }
    }

    println!("\nDone — commit bindings/ as build artifacts.");
}

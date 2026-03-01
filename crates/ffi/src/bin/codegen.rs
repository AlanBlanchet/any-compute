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
        .unwrap_or_else(|| PathBuf::from("out/bindings"));

    let registry = FfiRegistry::default_any_compute();

    println!(
        "Generating bindings for {} FFI functions...",
        registry.functions.len()
    );
    println!("Output directory: {}", out_dir.display());

    generate_all(&registry, &out_dir).expect("Failed to generate bindings");

    println!("\nGenerated:");
    println!("  Python:     {}/python/any_compute.py", out_dir.display());
    println!(
        "              {}/python/test_any_compute.py",
        out_dir.display()
    );
    println!(
        "  JavaScript: {}/javascript/any_compute.js",
        out_dir.display()
    );
    println!(
        "              {}/javascript/any_compute.d.ts",
        out_dir.display()
    );
    println!(
        "              {}/javascript/any_compute.test.js",
        out_dir.display()
    );
    println!(
        "  Java:       {}/java/com/anycompute/AnyCompute.java",
        out_dir.display()
    );
    println!(
        "              {}/java/com/anycompute/AnyComputeTest.java",
        out_dir.display()
    );
    println!("\nDone.");
}

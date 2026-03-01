/// Auto-resolve missing `-dev` symlinks for system libraries on Linux.
/// This means `libxdo-dev`, `libgtk-3-dev`, etc. don't need to be installed
/// as long as the runtime shared libraries are present.
fn main() {
    #[cfg(target_os = "linux")]
    {
        // libxdo: tao (used by dioxus desktop) links against -lxdo.
        // Often libxdo.so.3 exists but libxdo.so (dev symlink) does not.
        auto_symlink("libxdo.so.3", "libxdo.so");
    }
}

#[cfg(target_os = "linux")]
fn auto_symlink(from: &str, to: &str) {
    use std::path::Path;

    let search_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/lib/aarch64-linux-gnu",
    ];

    // Already available system-wide — nothing to do.
    for dir in &search_dirs {
        if Path::new(&format!("{dir}/{to}")).exists() {
            return;
        }
    }

    // Find the versioned .so and create a symlink in OUT_DIR.
    for dir in &search_dirs {
        let source = format!("{dir}/{from}");
        if Path::new(&source).exists() {
            let out_dir = std::env::var("OUT_DIR").unwrap();
            let link = format!("{out_dir}/{to}");
            let _ = std::os::unix::fs::symlink(&source, &link);
            println!("cargo:rustc-link-search=native={out_dir}");
            return;
        }
    }
}

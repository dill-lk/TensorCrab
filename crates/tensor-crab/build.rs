//! Build script for tensor-crab.
//!
//! When the `cuda` feature is enabled this script:
//! 1. Locates the CUDA Toolkit installation.
//! 2. Emits `cargo:rustc-link-lib` directives for `libcuda` and `libcudart`.
//! 3. Emits `cargo:rustc-link-search` directives for the CUDA library paths.
//!
//! The CUDA Toolkit root is resolved from (in priority order):
//! - `CUDA_PATH` environment variable  (Windows default)
//! - `CUDA_ROOT` environment variable
//! - `CUDA_HOME` environment variable
//! - `/usr/local/cuda`                 (Linux default)
//!
//! If CUDA is not found when the `cuda` feature is requested the build will
//! still succeed but you will get a linker error at link time explaining which
//! library is missing.

fn main() {
    // Re-run this script whenever these env vars change.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    #[cfg(feature = "cuda")]
    link_cuda();
}

/// Emits the linker flags needed to link against the CUDA Toolkit libraries.
#[cfg(feature = "cuda")]
fn link_cuda() {
    let cuda_root = find_cuda_root();

    // On Linux CUDA ships both lib/ and lib64/; on Windows it ships lib/x64/.
    let lib_dirs: &[&str] = if cfg!(target_os = "windows") {
        &["lib/x64"]
    } else {
        &["lib64", "lib"]
    };

    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}/{dir}", cuda_root);
    }

    // Link the CUDA Driver API (libcuda / cuda.lib) and Runtime API
    // (libcudart / cudart.lib).
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    // Link cuBLAS for GPU matrix multiplication.
    println!("cargo:rustc-link-lib=cublas");
}

/// Returns the path to the CUDA Toolkit root directory.
///
/// Searches environment variables in the order described in the module-level
/// doc comment, falling back to `/usr/local/cuda`.
#[cfg(feature = "cuda")]
fn find_cuda_root() -> String {
    // Try standard env vars first.
    for var in &["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME"] {
        if let Ok(val) = std::env::var(var) {
            if !val.is_empty() {
                return val;
            }
        }
    }
    // Default Linux installation path.
    "/usr/local/cuda".to_string()
}

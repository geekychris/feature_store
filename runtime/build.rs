use std::env;
use std::path::PathBuf;

fn main() {
    // ---- Proto compilation ----
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/scoring.proto"], &["proto/"])
        .expect("Failed to compile scoring.proto");

    // ---- C scoring library ----
    let generated_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .join("generated");

    cc::Build::new()
        .file("ffi/scoring_shim.c")
        .include(&generated_dir)
        .opt_level(2)
        .flag("-Wall")
        .flag("-Wno-unused-function") // header has many static inline fns
        .compile("scoring_cpu");

    println!("cargo:rustc-link-lib=static=scoring_cpu");
    println!("cargo:rerun-if-changed=ffi/scoring_shim.c");
    println!(
        "cargo:rerun-if-changed={}",
        generated_dir.join("scoring_split_core.h").display()
    );

    // ---- CUDA scoring library (optional) ----
    #[cfg(feature = "cuda")]
    {
        let cuda_shim = "ffi/scoring_cuda_shim.cu";
        if std::path::Path::new(cuda_shim).exists() {
            // Use nvcc directly — cc crate doesn't handle .cu files
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let status = std::process::Command::new("nvcc")
                .args([
                    "-O2",
                    "-c",
                    cuda_shim,
                    &format!("-I{}", generated_dir.display()),
                    "-o",
                    &format!("{}/scoring_cuda_shim.o", out_dir.display()),
                ])
                .status()
                .expect("Failed to run nvcc — is CUDA toolkit installed?");
            assert!(status.success(), "nvcc compilation failed");

            // Create static lib from object
            let status = std::process::Command::new("ar")
                .args([
                    "rcs",
                    &format!("{}/libscoring_cuda.a", out_dir.display()),
                    &format!("{}/scoring_cuda_shim.o", out_dir.display()),
                ])
                .status()
                .expect("Failed to run ar");
            assert!(status.success(), "ar failed");

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=scoring_cuda");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rerun-if-changed={}", cuda_shim);
        }
    }
}

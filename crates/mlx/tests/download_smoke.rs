//! Live HuggingFace download test. Requires network + HF_TOKEN.
//!
//!   MLX_DOWNLOAD_SMOKE=1 cargo test -p crabllm-mlx --test download_smoke -- --nocapture

use std::env;

fn skip_unless_enabled() -> bool {
    env::var("MLX_DOWNLOAD_SMOKE").ok().as_deref() != Some("1")
}

#[test]
fn download_small_model() {
    if skip_unless_enabled() {
        eprintln!("skipping: set MLX_DOWNLOAD_SMOKE=1 to run");
        return;
    }
    let dir = crabllm_mlx::download_model("mlx-community/SmolLM-135M-Instruct-4bit")
        .expect("download should succeed");
    assert!(dir.join("config.json").exists(), "config.json missing");
    assert!(
        std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().ends_with(".safetensors")),
        "no safetensors in {}",
        dir.display()
    );
}

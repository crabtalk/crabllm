//! Integration test for the HuggingFace downloader.
//!
//! Hits the real network — gated behind `MLX_DOWNLOAD_SMOKE=1` so CI
//! without internet access stays green. Pulls the smallest mlx-community
//! model I could find and verifies the file set is complete.
//!
//!   MLX_DOWNLOAD_SMOKE=1 cargo test -p crabllm-mlx --test download_smoke -- --nocapture
//!
//! Uses a throwaway tempdir so repeated runs are idempotent.

use crabllm_mlx::download;
use std::{env, fs, path::PathBuf};

// Smallest LLM on mlx-community at the time of writing. If this ever
// goes away, swap for another small `mlx-community/*` repo.
const TEST_REPO: &str = "mlx-community/SmolLM-135M-Instruct-4bit";

fn skip_unless_enabled() -> bool {
    env::var("MLX_DOWNLOAD_SMOKE").ok().as_deref() != Some("1")
}

fn tempdir() -> PathBuf {
    let base = env::temp_dir().join(format!("crabllm-mlx-download-smoke-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).expect("tempdir");
    base
}

#[test]
fn download_smolllm() {
    if skip_unless_enabled() {
        eprintln!("skipping download_smolllm: set MLX_DOWNLOAD_SMOKE=1 to run");
        return;
    }
    let cache = tempdir();
    let dir = download::download_model(TEST_REPO, &cache, &|_, _| {})
        .expect("download should succeed against live HF");
    assert!(dir.join("config.json").exists(), "config.json missing");
    assert!(
        dir.join("tokenizer_config.json").exists(),
        "tokenizer_config.json missing"
    );
    let has_weights = fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"));
    assert!(has_weights, "no .safetensors in {}", dir.display());

    // Second call should be a no-op (all files size-match).
    let again = download::download_model(TEST_REPO, &cache, &|_, _| {})
        .expect("second download should succeed");
    assert_eq!(dir, again);
}

#[test]
fn cached_path_before_download_returns_none() {
    let cache = tempdir();
    assert!(download::cached_model_path(TEST_REPO, &cache).is_none());
}

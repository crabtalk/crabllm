//! HuggingFace model downloader backed by the `hf-hub` crate.
//!
//! Delegates authentication (`$HF_TOKEN`), endpoint selection
//! (`$HF_ENDPOINT`), cache management, and file download to `hf-hub`.

use crabllm_core::Error;
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

/// File name suffixes matched by the wildcard part of the allowlist.
const ALLOWED_SUFFIXES: &[&str] = &[".safetensors", ".jinja"];

/// Exact filenames matched in addition to the suffix list.
const ALLOWED_EXACT: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.model",
    "model.safetensors.index.json",
];

fn build_api() -> Result<hf_hub::api::sync::Api, Error> {
    ApiBuilder::from_env()
        .build()
        .map_err(|e| Error::Internal(format!("mlx: failed to build HF API client: {e}")))
}

/// Check whether a model is already cached. Uses hf-hub's `get` which
/// returns the cached path without downloading if the file exists.
/// Returns `None` on any error (network misconfigured, not cached, etc.)
/// so callers fall through to a fresh download.
pub fn cached_model_path(repo: &str) -> Option<PathBuf> {
    // Cache::default() reads $HF_HOME but does not need network access.
    // We build a minimal Api just for the cache lookup — hf-hub's `get`
    // returns instantly for already-cached files.
    let api = build_api().ok()?;
    let repo_handle = api.model(repo.to_string());
    repo_handle
        .get("config.json")
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

/// Download all mlx-compatible files from a HuggingFace repo.
/// Returns the local snapshot directory.
///
/// Blocking. tokio callers must wrap in `spawn_blocking`.
pub fn download_model(repo: &str) -> Result<PathBuf, Error> {
    let api = build_api()?;
    let repo_handle = api.model(repo.to_string());

    tracing::info!(repo, "mlx: fetching model info");
    let info = repo_handle
        .info()
        .map_err(|e| Error::Internal(format!("mlx: model info for {repo}: {e}")))?;

    let wanted: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|name| is_wanted_filename(name))
        .collect();

    if wanted.is_empty() {
        return Err(Error::Internal(format!(
            "no mlx-compatible files in repo {repo}"
        )));
    }

    let mut model_dir: Option<PathBuf> = None;
    for filename in &wanted {
        tracing::info!(file = %filename, "mlx: downloading");
        let path = repo_handle
            .get(filename)
            .map_err(|e| Error::Internal(format!("mlx: download {filename}: {e}")))?;
        if model_dir.is_none() {
            model_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    model_dir.ok_or_else(|| Error::Internal("mlx: no files downloaded".to_string()))
}

fn is_wanted_filename(name: &str) -> bool {
    let basename = name.rsplit('/').next().unwrap_or(name);
    if ALLOWED_EXACT.iter().any(|exact| basename == *exact) {
        return true;
    }
    ALLOWED_SUFFIXES.iter().any(|ext| basename.ends_with(ext))
}

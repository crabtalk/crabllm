//! HuggingFace downloader for MLX models.
//!
//! Given a repo id like `mlx-community/Qwen3.5-0.8B-Instruct-4bit`,
//! materialize a directory containing the files mlx-swift-lm needs to
//! load the model: `config.json`, `tokenizer.json`,
//! `tokenizer_config.json`, any chat-template `.jinja` sidecars, and
//! every `*.safetensors` (sharded or not). This is the Rust side of
//! the "Swift takes a path, Rust handles the network" split.
//!
//! Concurrency / atomicity:
//!
//!   * Each file is downloaded to a `<name>.part.<pid>` tempfile in
//!     the same directory and atomically renamed onto its final name
//!     on success. Two processes racing on the same repo waste
//!     bandwidth but can never corrupt a half-written file a third
//!     reader might be about to mmap.
//!   * A sentinel file `.crabllm-mlx-complete` is written once every
//!     required file has landed. `cached_model_path` checks the
//!     sentinel, not `config.json` — a `config.json` alone means the
//!     cache is still mid-flight.
//!
//! Security:
//!
//!   * Repo ids are validated against a strict allowlist of characters
//!     (`[A-Za-z0-9._-]` per segment, exactly one slash) so a hostile
//!     value cannot escape the cache root via `..`, absolute paths,
//!     or NUL tricks.
//!   * Every filename in the HuggingFace API response is validated
//!     against the same allowlist plus a curated list of accepted
//!     basenames and suffixes. HF controls that field; we don't trust it.

use crabllm_core::Error;
use serde::Deserialize;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const HF_RESOLVE_BASE: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";
const USER_AGENT: &str = "crabllm-mlx";

const MAX_METADATA_SIZE: u64 = 4 * 1024 * 1024;
// Generous timeout: large safetensors may take 10+ minutes on a
// slow connection and the agent is shared across both metadata
// requests and blob downloads.
const AGENT_TIMEOUT: Duration = Duration::from_secs(60 * 30);
const READ_BUFFER: usize = 128 * 1024;

/// Sentinel file written inside the per-model cache directory once
/// every expected file has downloaded and renamed into place.
const COMPLETE_SENTINEL: &str = ".crabllm-mlx-complete";

/// File name suffixes matched by the wildcard part of the allowlist.
const ALLOWED_SUFFIXES: &[&str] = &[".safetensors", ".jinja"];

/// Exact filenames matched in addition to the suffix list. Keeps
/// `*.json` from matching random junk like `USE_POLICY.md.json`.
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

/// Progress callback: `(downloaded_so_far, total_known)`. `total_known`
/// is `None` while we are still learning file sizes from `Content-Length`
/// headers; after every file has had its GET issued it is `Some(sum)`.
pub type ProgressFn<'a> = &'a (dyn Fn(u64, Option<u64>) + Send + Sync);

/// `~/.crabtalk/cache/mlx/` — per-user cache root.
pub fn default_cache_dir() -> Result<PathBuf, Error> {
    dirs::home_dir()
        .map(|d| d.join(".crabtalk").join("cache").join("mlx"))
        .ok_or_else(|| Error::Internal("cannot determine home directory".to_string()))
}

/// Return the cached model directory for `repo` if the sentinel file
/// is present. A directory with `config.json` but no sentinel is still
/// mid-flight and will be treated as a cache miss.
pub fn cached_model_path(repo: &str, cache_dir: &Path) -> Option<PathBuf> {
    let path = cache_dir.join(repo);
    if path.join(COMPLETE_SENTINEL).exists() {
        Some(path)
    } else {
        None
    }
}

/// Fetch a model from `mlx-community` (or any HuggingFace repo with
/// the same layout) into `cache_dir`. Returns the per-model directory
/// path on success.
///
/// Blocking. tokio callers must wrap in `spawn_blocking`.
pub fn download_model(
    repo: &str,
    cache_dir: &Path,
    on_progress: ProgressFn<'_>,
) -> Result<PathBuf, Error> {
    validate_repo_id(repo)?;
    let model_dir = cache_dir.join(repo);
    fs::create_dir_all(&model_dir).map_err(|e| {
        Error::Internal(format!(
            "failed to create cache dir {}: {e}",
            model_dir.display()
        ))
    })?;

    // Already complete — cheapest possible path.
    if model_dir.join(COMPLETE_SENTINEL).exists() {
        return Ok(model_dir);
    }

    let agent = build_agent(AGENT_TIMEOUT);

    tracing::info!(repo = repo, "mlx: fetching model index");
    let info = fetch_model_info(&agent, repo)?;

    let wanted: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|name| is_wanted_filename(name))
        .collect();
    if wanted.is_empty() {
        return Err(Error::Internal(format!(
            "no mlx-compatible files (safetensors/json/jinja/tokenizer.model) in repo {repo}"
        )));
    }

    let mut downloaded_total: u64 = 0;
    let mut known_sizes_sum: u64 = 0;
    for (idx, filename) in wanted.iter().enumerate() {
        validate_filename(filename)?;
        let dest = model_dir.join(filename);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::Internal(format!("mkdir {}: {e}", parent.display())))?;
        }

        let size = download_file(&agent, repo, filename, &dest, &mut |delta| {
            downloaded_total += delta;
            // Total is only "known" once every file through this one
            // has finished and we've summed their Content-Lengths.
            on_progress(downloaded_total, None);
        })?;

        known_sizes_sum += size;
        let is_last = idx + 1 == wanted.len();
        if is_last {
            on_progress(downloaded_total, Some(known_sizes_sum));
        }
    }

    // Mark the cache complete. Do this *after* every file has landed
    // so a crashed download never leaves a false-positive sentinel.
    fs::write(model_dir.join(COMPLETE_SENTINEL), b"")
        .map_err(|e| Error::Internal(format!("write sentinel: {e}")))?;

    Ok(model_dir)
}

// ---------- validation ----------

fn validate_repo_id(repo: &str) -> Result<(), Error> {
    if repo.is_empty() {
        return Err(Error::Internal(
            "invalid model repo id: empty (must be namespace/name)".to_string(),
        ));
    }
    // Exactly one slash dividing two segments (HF namespace/name layout).
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 {
        return Err(Error::Internal(format!(
            "invalid model repo id: {repo:?} (must be namespace/name)"
        )));
    }
    validate_path_segment(parts[0], "repo namespace")?;
    validate_path_segment(parts[1], "repo name")?;
    Ok(())
}

fn validate_filename(name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::Internal("empty filename in HF listing".to_string()));
    }
    for segment in name.split('/') {
        validate_path_segment(segment, "filename segment")?;
    }
    Ok(())
}

fn validate_path_segment(segment: &str, what: &str) -> Result<(), Error> {
    if segment.is_empty() || segment == "." || segment == ".." {
        return Err(Error::Internal(format!(
            "invalid model repo id: bad {what} {segment:?}"
        )));
    }
    for c in segment.chars() {
        let ok = c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
        if !ok {
            return Err(Error::Internal(format!(
                "invalid model repo id: bad {what} {segment:?} (char {c:?})"
            )));
        }
    }
    Ok(())
}

pub(crate) fn is_wanted_filename(name: &str) -> bool {
    // Match the basename — nothing in mlx-community nests files in a
    // way that matters for the allowlist, but defend against it anyway.
    let basename = name.rsplit('/').next().unwrap_or(name);
    if ALLOWED_EXACT.iter().any(|exact| basename == *exact) {
        return true;
    }
    ALLOWED_SUFFIXES.iter().any(|ext| basename.ends_with(ext))
}

// ---------- HTTP ----------

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
}

fn build_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(timeout))
            .build(),
    )
}

fn fetch_model_info(agent: &ureq::Agent, repo: &str) -> Result<ModelInfo, Error> {
    let url = format!("{HF_API_BASE}/{repo}");
    let resp = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| Error::Internal(format!("HTTP failed for {url}: {e}")))?;
    let body = resp
        .into_body()
        .with_config()
        .limit(MAX_METADATA_SIZE)
        .read_to_string()
        .map_err(|e| Error::Internal(format!("read {url}: {e}")))?;
    serde_json::from_str(&body).map_err(|e| Error::Internal(format!("parse {url}: {e}")))
}

/// Download one file with optional resume. Writes to a per-pid
/// `.part.<pid>` tempfile and renames on success so concurrent
/// downloaders never corrupt each other's output. Returns the final
/// file size on disk.
fn download_file(
    agent: &ureq::Agent,
    repo: &str,
    filename: &str,
    dest: &Path,
    on_chunk: &mut dyn FnMut(u64),
) -> Result<u64, Error> {
    let url = format!("{HF_RESOLVE_BASE}/{repo}/resolve/{DEFAULT_REVISION}/{filename}");

    // Per-pid tempfile sits next to the final destination.
    let parent = dest
        .parent()
        .ok_or_else(|| Error::Internal(format!("dest {} has no parent", dest.display())))?;
    let basename = dest
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Internal(format!("dest {} has no basename", dest.display())))?;
    let partial = parent.join(format!("{basename}.part.{}", std::process::id()));
    let partial_size = partial.metadata().map(|m| m.len()).unwrap_or(0);

    let mut req = agent.get(&url).header("User-Agent", USER_AGENT);
    if partial_size > 0 {
        req = req.header("Range", &format!("bytes={partial_size}-"));
    }
    let resp = req
        .call()
        .map_err(|e| Error::Internal(format!("download {filename}: {e}")))?;

    let append = resp.status().as_u16() == 206;
    if !append && partial_size > 0 {
        // Server refused the range; discard the partial so we don't
        // append onto a stale prefix.
        let _ = fs::remove_file(&partial);
    }

    // Content-Length for a 206 response is the length of the *remaining*
    // body, not the full file. Rather than parse Content-Range we
    // remember what we had on disk and add it back at the end.
    let remaining = parse_content_length(&resp);
    let expected_total = remaining.map(|n| if append { partial_size + n } else { n });

    let mut file = if append {
        fs::OpenOptions::new().append(true).open(&partial)
    } else {
        fs::File::create(&partial)
    }
    .map_err(|e| Error::Internal(format!("open {}: {e}", partial.display())))?;

    // Cap the stream at expected + 1 MiB of slack so a runaway server
    // can't fill the disk. If Content-Length was absent, fall back to
    // a hard 32 GiB ceiling — much larger than any MLX weight file
    // we'd plausibly fetch (Qwen 3.5 27B 4-bit is ~17 GiB).
    let cap = expected_total
        .map(|n| n.saturating_add(1024 * 1024))
        .unwrap_or(32 * 1024 * 1024 * 1024);

    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; READ_BUFFER];
    let mut this_file_bytes: u64 = 0;
    loop {
        if append_and_partial_exceeds(append, partial_size, this_file_bytes, cap) {
            return Err(Error::Internal(format!(
                "download {filename}: exceeded cap of {cap} bytes"
            )));
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::Internal(format!("read {filename}: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| Error::Internal(format!("write {}: {e}", partial.display())))?;
        this_file_bytes += n as u64;
        on_chunk(n as u64);
    }
    file.flush()
        .map_err(|e| Error::Internal(format!("flush {}: {e}", partial.display())))?;
    drop(file);

    let final_size = partial
        .metadata()
        .map(|m| m.len())
        .map_err(|e| Error::Internal(format!("stat {}: {e}", partial.display())))?;

    if let Some(expected) = expected_total {
        if final_size != expected {
            let _ = fs::remove_file(&partial);
            return Err(Error::Internal(format!(
                "size mismatch for {filename}: expected {expected}, got {final_size}"
            )));
        }
    }

    // Atomic rename. On Unix this is a single syscall; on Windows we'd
    // need `rename` with the REPLACE_EXISTING flag — we're macOS/iOS
    // only so this is fine.
    fs::rename(&partial, dest).map_err(|e| {
        Error::Internal(format!(
            "rename {} -> {}: {e}",
            partial.display(),
            dest.display()
        ))
    })?;

    Ok(final_size)
}

fn append_and_partial_exceeds(append: bool, partial: u64, body: u64, cap: u64) -> bool {
    let total = if append { partial + body } else { body };
    total > cap
}

fn parse_content_length(resp: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    resp.headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
}

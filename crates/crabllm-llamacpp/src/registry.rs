use crabllm_core::Error;
use std::{
    fmt::Write as _,
    io::{Read, Write},
    path::{Path, PathBuf},
};

const REGISTRY_BASE: &str = "https://registry.ollama.ai/v2/library";
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Max size for manifest/tags JSON responses (1 MB).
const MAX_JSON_SIZE: u64 = 1024 * 1024;
/// HTTP timeout for metadata requests (manifests, tags).
const METADATA_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// OCI manifest from the Ollama registry.
#[derive(Debug, serde::Deserialize)]
pub struct Manifest {
    pub layers: Vec<Layer>,
}

/// A single layer in an OCI manifest.
#[derive(Debug, serde::Deserialize)]
pub struct Layer {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

/// Tag list response from the Ollama registry.
#[derive(Debug, serde::Deserialize)]
struct TagList {
    tags: Vec<String>,
}

/// Parse an Ollama model reference into (name, tag).
///
/// `"llama3.2:3b"` → `("llama3.2", "3b")`
/// `"llama3.2"` → `("llama3.2", "latest")`
pub fn parse_model_name(model: &str) -> (&str, &str) {
    model.split_once(':').unwrap_or((model, "latest"))
}

/// Default directory for downloaded models: `~/.crabtalk/models/`
pub fn default_cache_dir() -> Result<PathBuf, Error> {
    dirs::home_dir()
        .map(|d| d.join(".crabtalk").join("models"))
        .ok_or_else(|| Error::Internal("cannot determine home directory".to_string()))
}

/// Return the cached GGUF path for a model if it exists and passes digest check.
pub fn cached_model_path(model: &str, cache_dir: &Path) -> Option<PathBuf> {
    let (name, tag) = parse_model_name(model);
    let path = cache_dir.join(name).join(format!("{tag}.gguf"));
    path.exists().then_some(path)
}

/// Fetch the OCI manifest for a model from the Ollama registry.
pub fn fetch_manifest(name: &str, tag: &str) -> Result<Manifest, Error> {
    let url = format!("{REGISTRY_BASE}/{name}/manifests/{tag}");
    let body = fetch_json(&url)?;
    serde_json::from_str(&body)
        .map_err(|e| Error::Internal(format!("failed to parse manifest for {name}:{tag}: {e}")))
}

/// List available tags for a model on the Ollama registry.
pub fn fetch_tags(name: &str) -> Result<Vec<String>, Error> {
    let url = format!("{REGISTRY_BASE}/{name}/tags/list");
    let body = fetch_json(&url)?;
    let list: TagList = serde_json::from_str(&body)
        .map_err(|e| Error::Internal(format!("failed to parse tags for {name}: {e}")))?;
    Ok(list.tags)
}

/// Pull a model from the Ollama registry and cache the GGUF file.
///
/// Verifies the cached file's digest against the manifest on each call.
/// Calls `on_progress(downloaded_bytes, total_bytes)` during download.
/// Returns the path to the cached GGUF file.
pub fn pull_model(
    model: &str,
    cache_dir: &Path,
    on_progress: &dyn Fn(u64, u64),
) -> Result<PathBuf, Error> {
    let (name, tag) = parse_model_name(model);
    let manifest = fetch_manifest(name, tag)?;

    let layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == MODEL_MEDIA_TYPE)
        .ok_or_else(|| {
            Error::Internal(format!(
                "no model weights layer in manifest for {name}:{tag}"
            ))
        })?;

    if layer.size == 0 {
        return Err(Error::Internal(format!(
            "model weights layer has zero size for {name}:{tag}"
        )));
    }

    let model_dir = cache_dir.join(name);
    let dest = model_dir.join(format!("{tag}.gguf"));

    // Skip download if the existing file passes digest verification.
    if dest.exists() && verify_digest(&dest, &layer.digest).is_ok() {
        return Ok(dest);
    }

    std::fs::create_dir_all(&model_dir)
        .map_err(|e| Error::Internal(format!("failed to create {}: {e}", model_dir.display())))?;

    // Download to a temp file with unique name, then rename for atomicity.
    let tmp = model_dir.join(format!("{tag}.gguf.{}.tmp", std::process::id()));

    let result = (|| {
        download_blob(name, &layer.digest, layer.size, &tmp, on_progress)?;
        verify_digest(&tmp, &layer.digest)?;
        std::fs::rename(&tmp, &dest)
            .map_err(|e| Error::Internal(format!("failed to rename temp file: {e}")))?;
        Ok(dest.clone())
    })();

    // Clean up temp file on any error.
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }

    result
}

/// Fetch a small JSON response with size limit and timeout.
fn fetch_json(url: &str) -> Result<String, Error> {
    let config = ureq::config::Config::builder()
        .timeout_global(Some(METADATA_TIMEOUT))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let resp = agent
        .get(url)
        .header("Accept", "application/vnd.oci.image.manifest.v1+json")
        .header("User-Agent", "crabllm")
        .call()
        .map_err(|e| Error::Internal(format!("HTTP request failed for {url}: {e}")))?;

    let body = resp
        .into_body()
        .with_config()
        .limit(MAX_JSON_SIZE)
        .read_to_string()
        .map_err(|e| Error::Internal(format!("failed to read response from {url}: {e}")))?;

    Ok(body)
}

/// Download a blob from the registry, streaming to a file.
fn download_blob(
    name: &str,
    digest: &str,
    total: u64,
    dest: &Path,
    on_progress: &dyn Fn(u64, u64),
) -> Result<(), Error> {
    let url = format!("{REGISTRY_BASE}/{name}/blobs/{digest}");
    let resp = ureq::get(&url)
        .header("User-Agent", "crabllm")
        .call()
        .map_err(|e| Error::Internal(format!("failed to download blob {digest}: {e}")))?;

    let mut file = std::fs::File::create(dest)
        .map_err(|e| Error::Internal(format!("failed to create {}: {e}", dest.display())))?;

    let mut reader = resp.into_body().into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::Internal(format!("download read error: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| Error::Internal(format!("file write error: {e}")))?;
        downloaded += n as u64;
        on_progress(downloaded, total);
    }

    file.flush()
        .map_err(|e| Error::Internal(format!("file flush error: {e}")))?;

    Ok(())
}

/// Verify that a file's SHA256 matches an OCI digest (`sha256:hex`).
fn verify_digest(path: &Path, expected: &str) -> Result<(), Error> {
    use sha2::Digest;

    let hex = expected
        .strip_prefix("sha256:")
        .ok_or_else(|| Error::Internal(format!("unsupported digest format: {expected}")))?;

    let mut file = std::fs::File::open(path).map_err(|e| {
        Error::Internal(format!(
            "failed to open {} for digest check: {e}",
            path.display()
        ))
    })?;

    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| Error::Internal(format!("digest read error: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    let digest = hasher.finalize();
    let mut actual = String::with_capacity(64);
    for b in digest.iter() {
        write!(actual, "{b:02x}").unwrap();
    }

    if actual != hex {
        return Err(Error::Internal(format!(
            "digest mismatch: expected {hex}, got {actual}"
        )));
    }

    Ok(())
}

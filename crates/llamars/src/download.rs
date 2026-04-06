use crate::BINARY_NAME;
use crabllm_core::Error;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Detected GPU backend for selecting the correct llama-server build.
#[derive(Debug)]
enum GpuBackend {
    /// No GPU acceleration — CPU only.
    Cpu,
    /// macOS Metal — always available, built into the standard macOS binary.
    Metal,
    /// NVIDIA CUDA with detected toolkit version (e.g. "12.4").
    Cuda { version: String },
}

/// Default directory for downloaded llama-server binaries: `~/.crabtalk/bin/llamacpp`
pub fn install_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".crabtalk")
        .join("bin")
        .join("llamacpp")
}

/// Detect the GPU backend available on this machine.
fn detect_gpu() -> GpuBackend {
    if cfg!(target_os = "macos") {
        return GpuBackend::Metal;
    }

    // Check for NVIDIA GPU via nvidia-smi.
    if let Ok(output) = Command::new("nvidia-smi").output()
        && output.status.success()
        && let Some(version) = parse_cuda_version(&output.stdout)
    {
        tracing::info!(cuda_version = %version, "detected NVIDIA CUDA");
        return GpuBackend::Cuda { version };
    }

    GpuBackend::Cpu
}

/// Parse CUDA version from nvidia-smi output.
///
/// Looks for "CUDA Version: X.Y" in the output.
fn parse_cuda_version(output: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(output).ok()?;
    let marker = "CUDA Version: ";
    let start = text.find(marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.')?;
    let version = &rest[..end];
    if version.contains('.') {
        Some(version.to_string())
    } else {
        None
    }
}

/// Determine the best release asset name for this platform and GPU.
///
/// Fetches the release asset list and picks the best match for our
/// OS, architecture, and GPU backend.
fn pick_asset(tag: &str, assets: &[String], gpu: &GpuBackend) -> Result<String, Error> {
    let arch_part = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => {
            return Err(Error::Internal(format!(
                "unsupported architecture: {other}"
            )));
        }
    };

    let os = std::env::consts::OS;

    // Build candidate patterns in priority order.
    let candidates: Vec<String> = match (os, gpu) {
        ("macos", _) => {
            vec![format!("llama-{tag}-bin-macos-{arch_part}")]
        }
        ("linux", GpuBackend::Cuda { version }) => {
            let mut c = cuda_candidates(tag, version, "ubuntu", arch_part);
            // Fallback to CPU if no CUDA asset matches.
            c.push(format!("llama-{tag}-bin-ubuntu-{arch_part}"));
            c
        }
        ("linux", _) => {
            vec![format!("llama-{tag}-bin-ubuntu-{arch_part}")]
        }
        ("windows", GpuBackend::Cuda { version }) => {
            let mut c = cuda_candidates(tag, version, "win-cuda", arch_part);
            c.push(format!("llama-{tag}-bin-win-cpu-{arch_part}"));
            c
        }
        ("windows", _) => {
            vec![format!("llama-{tag}-bin-win-cpu-{arch_part}")]
        }
        _ => {
            return Err(Error::Internal(format!("unsupported OS: {os}")));
        }
    };

    // Find the first candidate that matches an actual release asset.
    for candidate in &candidates {
        if let Some(asset) = assets.iter().find(|a| a.starts_with(candidate)) {
            return Ok(asset.clone());
        }
    }

    Err(Error::Internal(format!(
        "no matching release asset for {os}/{arch_part} (tag {tag})"
    )))
}

/// Generate CUDA asset name candidates, preferring the highest compatible version.
///
/// Given detected CUDA 12.6, generates candidates like:
/// - `llama-{tag}-bin-cuda-cu12.6...`  (exact match)
/// - `llama-{tag}-bin-cuda-cu12.4...`  (compatible, lower minor)
/// - `llama-{tag}-bin-cuda-cu12.2...`  (compatible, even lower)
///   CUDA is forward-compatible within the same major version.
fn cuda_candidates(tag: &str, cuda_version: &str, os_part: &str, arch: &str) -> Vec<String> {
    let parts: Vec<&str> = cuda_version.split('.').collect();
    let major = parts.first().copied().unwrap_or("12");
    let minor: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Generate candidates from detected version down to 0.
    (0..=minor)
        .rev()
        .map(|m| format!("llama-{tag}-bin-{os_part}-cu{major}.{m}"))
        .chain(
            // Also try just the major version pattern.
            std::iter::once(format!("llama-{tag}-bin-{os_part}-cu{major}")),
        )
        // On Linux, CUDA assets use "cuda" prefix, not "ubuntu".
        .chain(
            (0..=minor)
                .rev()
                .map(|m| format!("llama-{tag}-bin-cuda-cu{major}.{m}-ubuntu-{arch}")),
        )
        .collect()
}

/// Fetch release info from the GitHub API. Returns (tag, asset_names).
fn fetch_release(tag: Option<&str>) -> Result<(String, Vec<String>), Error> {
    let url = match tag {
        Some(t) => format!("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{t}"),
        None => "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest".to_string(),
    };

    let resp = ureq::get(&url)
        .header("User-Agent", "llamars")
        .call()
        .map_err(|e| Error::Internal(format!("failed to fetch release: {e}")))?;

    let body = resp
        .into_body()
        .read_to_string()
        .map_err(|e| Error::Internal(format!("failed to read release response: {e}")))?;

    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| Error::Internal(format!("failed to parse release JSON: {e}")))?;

    let tag_name = json["tag_name"]
        .as_str()
        .ok_or_else(|| Error::Internal("missing tag_name in release".to_string()))?
        .to_string();

    let assets = json["assets"]
        .as_array()
        .ok_or_else(|| Error::Internal("missing assets in release".to_string()))?
        .iter()
        .filter_map(|a| a["name"].as_str().map(|s| s.to_string()))
        .collect();

    Ok((tag_name, assets))
}

/// Download the llama-server binary for the current platform.
///
/// Detects the GPU backend (CUDA, Metal, or CPU) and downloads the
/// matching build from `ggml-org/llama.cpp` GitHub releases.
///
/// Returns the path to the installed binary.
pub fn download(tag: Option<&str>) -> Result<PathBuf, Error> {
    tracing::info!("fetching llama.cpp release...");
    let (tag, assets) = fetch_release(tag)?;

    let gpu = detect_gpu();
    let asset = pick_asset(&tag, &assets, &gpu)?;

    tracing::info!(asset = %asset, gpu = ?gpu, "downloading llama-server");

    let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}");
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| Error::Internal(format!("download failed: {e}")))?;

    let body = resp
        .into_body()
        .read_to_vec()
        .map_err(|e| Error::Internal(format!("failed to read download body: {e}")))?;

    let dir = install_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::Internal(format!("failed to create {}: {e}", dir.display())))?;

    let dest = dir.join(BINARY_NAME);

    if asset.ends_with(".tar.gz") {
        extract_tar_gz(&body, BINARY_NAME, &dest)?;
    } else if asset.ends_with(".zip") {
        extract_zip(&body, BINARY_NAME, &dest)?;
    } else {
        return Err(Error::Internal(format!("unknown archive format: {asset}")));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&dest, perms)
            .map_err(|e| Error::Internal(format!("failed to chmod: {e}")))?;
    }

    tracing::info!(path = %dest.display(), "installed llama-server");
    Ok(dest)
}

/// Extract a specific binary from a .tar.gz archive.
fn extract_tar_gz(data: &[u8], binary_name: &str, dest: &Path) -> Result<(), Error> {
    use std::io::Cursor;

    let decoder = flate2::read::GzDecoder::new(Cursor::new(data));
    let mut archive = tar::Archive::new(decoder);

    for entry in archive
        .entries()
        .map_err(|e| Error::Internal(format!("failed to read tar entries: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| Error::Internal(format!("failed to read tar entry: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| Error::Internal(format!("invalid tar entry path: {e}")))?;

        if path.file_name().and_then(|f| f.to_str()) == Some(binary_name) {
            let mut out = std::fs::File::create(dest).map_err(|e| {
                Error::Internal(format!("failed to create {}: {e}", dest.display()))
            })?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| Error::Internal(format!("failed to extract {binary_name}: {e}")))?;
            return Ok(());
        }
    }

    Err(Error::Internal(format!(
        "{binary_name} not found in archive"
    )))
}

/// Extract a specific binary from a .zip archive.
fn extract_zip(data: &[u8], binary_name: &str, dest: &Path) -> Result<(), Error> {
    use std::io::Cursor;

    let mut archive = zip::ZipArchive::new(Cursor::new(data))
        .map_err(|e| Error::Internal(format!("failed to open zip: {e}")))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| Error::Internal(format!("failed to read zip entry: {e}")))?;

        let name = file.name().to_string();
        let entry_path = Path::new(&name);
        if entry_path.file_name().and_then(|f| f.to_str()) == Some(binary_name)
            && !name.ends_with('/')
        {
            let mut out = std::fs::File::create(dest).map_err(|e| {
                Error::Internal(format!("failed to create {}: {e}", dest.display()))
            })?;
            std::io::copy(&mut file, &mut out)
                .map_err(|e| Error::Internal(format!("failed to extract {binary_name}: {e}")))?;
            return Ok(());
        }
    }

    Err(Error::Internal(format!(
        "{binary_name} not found in archive"
    )))
}

use crate::BINARY_NAME;
use crabllm_core::Error;
use std::path::{Path, PathBuf};

/// Default directory for downloaded llama-server binaries.
///
/// Resolution order:
/// 1. `$CRABLLM_HOME/bin`
/// 2. Platform data directory:
///    - Linux:   `~/.local/share/crabllm/bin`
///    - macOS:   `~/Library/Application Support/crabllm/bin`
///    - Windows: `%LOCALAPPDATA%\crabllm\bin`
pub fn install_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CRABLLM_HOME") {
        return PathBuf::from(dir).join("bin");
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("crabllm")
        .join("bin")
}

/// Determine the release asset name for the current platform.
///
/// Maps (os, arch) to the llama.cpp release naming convention:
/// - linux x86_64   → `llama-{tag}-bin-ubuntu-x64.tar.gz`
/// - linux aarch64   → `llama-{tag}-bin-ubuntu-arm64.tar.gz`
/// - macos x86_64   → `llama-{tag}-bin-macos-x64.tar.gz`
/// - macos aarch64   → `llama-{tag}-bin-macos-arm64.tar.gz`
/// - windows x86_64  → `llama-{tag}-bin-win-cpu-x64.zip`
/// - windows aarch64  → `llama-{tag}-bin-win-cpu-arm64.zip`
fn asset_name(tag: &str) -> Result<String, Error> {
    let (os_part, ext) = match std::env::consts::OS {
        "linux" => ("ubuntu", "tar.gz"),
        "macos" => ("macos", "tar.gz"),
        "windows" => ("win-cpu", "zip"),
        other => {
            return Err(Error::Internal(format!(
                "unsupported OS for llama-server download: {other}"
            )));
        }
    };

    let arch_part = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => {
            return Err(Error::Internal(format!(
                "unsupported architecture for llama-server download: {other}"
            )));
        }
    };

    Ok(format!("llama-{tag}-bin-{os_part}-{arch_part}.{ext}"))
}

/// Fetch the latest release tag from the GitHub API.
fn fetch_latest_tag() -> Result<String, Error> {
    let resp = ureq::get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
        .header("User-Agent", "crabllm")
        .call()
        .map_err(|e| Error::Internal(format!("failed to fetch latest release: {e}")))?;

    let body_str = resp
        .into_body()
        .read_to_string()
        .map_err(|e| Error::Internal(format!("failed to read release response: {e}")))?;
    let body: serde_json::Value = serde_json::from_str(&body_str)
        .map_err(|e| Error::Internal(format!("failed to parse release JSON: {e}")))?;

    body["tag_name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Internal("missing tag_name in release response".to_string()))
}

/// Download the llama-server binary for the current platform.
///
/// Downloads from `ggml-org/llama.cpp` GitHub releases, extracts the
/// archive, and places the `llama-server` binary in the install directory.
///
/// Returns the path to the installed binary.
pub fn download(tag: Option<&str>) -> Result<PathBuf, Error> {
    let tag = match tag {
        Some(t) => t.to_string(),
        None => {
            eprintln!("fetching latest llama.cpp release...");
            fetch_latest_tag()?
        }
    };

    let asset = asset_name(&tag)?;
    let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}");

    eprintln!("downloading {asset}...");

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

    // Make executable on unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&dest, perms)
            .map_err(|e| Error::Internal(format!("failed to chmod: {e}")))?;
    }

    eprintln!("installed llama-server to {}", dest.display());
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

        // The binary may be at the top level or inside a subdirectory.
        // Use file_name() which handles both `/` and `\` path separators.
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
        // Zip entries may use `/` or `\` as separators. Convert to Path
        // for cross-platform file_name() matching.
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

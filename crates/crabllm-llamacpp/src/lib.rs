//! crabllm-llamacpp — managed llama.cpp server with Ollama registry support.
//!
//! This crate handles the full lifecycle of a `llama-server` binary:
//! finding it on disk, downloading it from GitHub releases, spawning
//! the process, health-checking, tearing it down on drop, and fetching
//! models from the Ollama registry.

use crabllm_core::Error;
pub use download::{download, install_dir};
pub use pool::ServerPool;
pub use server::{LlamaCppConfig, LlamaCppServer};
use std::path::PathBuf;

mod download;
pub mod pool;
pub mod proxy;
pub mod registry;
mod server;

/// The platform-specific binary name for llama-server.
pub const BINARY_NAME: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};

/// Find the `llama-server` binary, auto-downloading if not found.
///
/// Search order:
/// 1. `$LLAMA_SERVER` environment variable
/// 2. `llama-server` on `$PATH`
/// 3. Default install directory
/// 4. Auto-download from GitHub releases (detects GPU backend)
pub fn find_server_binary() -> Result<PathBuf, Error> {
    if let Ok(path) = std::env::var("LLAMA_SERVER") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
        return Err(Error::Internal(format!(
            "LLAMA_SERVER={path} does not exist"
        )));
    }

    if let Ok(p) = which::which("llama-server") {
        return Ok(p);
    }

    let installed = install_dir().join(BINARY_NAME);
    if installed.exists() {
        return Ok(installed);
    }

    // Not found anywhere — auto-download the correct build.
    tracing::info!("llama-server not found, downloading...");
    download(None)
}

//! Managed llama.cpp server for crabllm.
//!
//! This crate handles the full lifecycle of a `llama-server` binary:
//! finding it on disk, downloading it from GitHub releases, spawning
//! the process, health-checking, and tearing it down on drop.

use crabllm_core::Error;
pub use download::{download, install_dir};
pub use server::{LlamaCppConfig, LlamaCppServer};
use std::path::PathBuf;

mod download;
mod server;

/// The platform-specific binary name for llama-server.
pub const BINARY_NAME: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};

/// Find the `llama-server` binary.
///
/// Search order:
/// 1. `$LLAMA_SERVER` environment variable
/// 2. `llama-server` on `$PATH`
/// 3. crabllm's default install directory
///    - `$CRABLLM_HOME/bin`
///    - or platform default: Linux `~/.local/share/crabllm/bin`,
///      macOS `~/Library/Application Support/crabllm/bin`,
///      Windows `%LOCALAPPDATA%\crabllm\bin`
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

    Err(Error::Internal(
        "llama-server not found. Run `crabllm llamacpp download` to install it".to_string(),
    ))
}

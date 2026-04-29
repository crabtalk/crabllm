//! crabllm-mlx — MLX local inference provider for the crabllm gateway.
//!
//! This crate sits between the Rust gateway and a Swift static library
//! built from the sibling `mlx/` directory at the repo root. The Swift
//! side links against `mlx-swift-lm` (main branch) and exposes a C ABI
//! declared in `mlx/include/crabllm_mlx.h`.
//!
//! Public surface (Apple targets only):
//!
//!   * [`MlxPool`] — Swift-managed multi-model cache with idle
//!     eviction. Handles on-demand model loading internally.
//!   * [`MlxProvider`] — `Provider`-trait front door that resolves
//!     model names (path or HuggingFace repo), delegates to the pool,
//!     and reassembles OpenAI-shape responses.
//!
//! Only macOS and iOS are supported — MLX is Apple-Silicon-only. On
//! every other target the crate still compiles (so `cargo build` on
//! Linux CI passes) but every entry point returns
//! `Error::not_implemented`.

pub mod download;
pub mod registry;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod ffi;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metallib;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod pool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod provider;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod session;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod gemma_patch;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use pool::MlxPool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use provider::MlxProvider;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod stub;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use stub::{MlxPool, MlxProvider};

pub use download::{DownloadEvent, cached_model_path, download_model};

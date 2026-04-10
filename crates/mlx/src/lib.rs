//! crabllm-mlx — MLX local inference provider for the crabllm gateway.
//!
//! This crate sits between the Rust gateway and a Swift static library
//! built from the sibling `mlx/` directory at the repo root. The Swift
//! side (see `mlx/Sources/CrabllmMlx`) links against `mlx-swift-lm`
//! and exposes a small C ABI declared in `mlx/include/crabllm_mlx.h`.
//!
//! Public surface (Apple targets only):
//!
//!   * [`Session`] — low-level safe wrapper over the C ABI. One
//!     loaded model. Reentrant. Used directly by callers who want to
//!     skip the OpenAI-shape pipeline.
//!   * [`MlxModel`] — `Provider`-trait wrapper around a single
//!     [`Session`], with HuggingFace download integration.
//!   * [`MlxPool`] — multi-model on-demand cache with idle eviction.
//!   * [`MlxProvider`] — `Provider`-trait front door that dispatches
//!     by `request.model` through an [`MlxPool`].
//!
//! Only macOS and iOS are supported — MLX is Apple-Silicon-only. On
//! every other target the crate still compiles (so `cargo build` on
//! Linux CI passes) but every entry point returns
//! `Error::not_implemented`.

pub mod download;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod ffi;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod model;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod pool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod provider;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod session;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use model::MlxModel;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use pool::MlxPool;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use provider::MlxProvider;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use session::{GenerateOptions, GenerateOutput, GenerateRequest, Session, StreamOutput};

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod stub;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use stub::{
    GenerateOptions, GenerateOutput, GenerateRequest, MlxModel, MlxPool, MlxProvider, Session,
    StreamOutput,
};

pub use download::{cached_model_path, default_cache_dir, download_model};

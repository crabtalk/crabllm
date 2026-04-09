//! crabllm-mlx — MLX local inference provider for the crabllm gateway.
//!
//! This crate sits between the Rust gateway and a Swift static library
//! built from the sibling `mlx/` directory at the repo root. The Swift
//! side (see `mlx/Sources/CrabllmMlx`) links against `mlx-swift-lm`
//! and exposes a small C ABI declared in `mlx/include/crabllm_mlx.h`.
//!
//! Only macOS and iOS are supported — MLX is Apple-Silicon-only. On
//! every other target the crate still compiles (so `cargo build` on
//! Linux CI passes) but every entry point returns
//! `Error::not_implemented`.

pub mod download;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod ffi;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod session;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use session::{GenerateOptions, GenerateOutput, GenerateRequest, Session, StreamOutput};

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod stub;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub use stub::{GenerateOptions, GenerateOutput, GenerateRequest, Session, StreamOutput};

pub use download::{cached_model_path, default_cache_dir, download_model};

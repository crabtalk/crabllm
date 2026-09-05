#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub use config::{
    GatewayConfig, KeyConfig, KeyRateLimit, ProviderConfig, ProviderKind, StorageConfig,
};
pub use error::{ApiError, ApiErrorBody, Error};
#[cfg(feature = "std")]
pub use extension::{Extension, ExtensionError, RequestContext};
pub use model_info::ModelInfo;
pub use provider::{BoxStream, ByteStream, Provider};
#[cfg(feature = "std")]
pub use retrying::Retrying;
pub use storage::{BoxFuture, KvPairs, PREFIX_LEN, Prefix, Storage, storage_key};
pub use types::{anthropic, gemini, ir, *};
pub use usage::Usage;

pub mod codec;
#[cfg(feature = "std")]
mod config;
mod error;
#[cfg(feature = "std")]
mod extension;
pub mod json;
mod model_info;
#[cfg(feature = "openapi")]
mod openapi;
mod provider;
#[cfg(feature = "std")]
mod retrying;
mod storage;
mod types;
mod usage;

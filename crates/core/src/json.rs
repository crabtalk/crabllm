//! JSON ser/de surface — `to_vec` / `from_slice` / `to_string` / `from_str`.
//! With the `sonic` feature these route to SIMD-accelerated `sonic-rs`;
//! without it, to `serde_json`.
//!
//! For `Value`, `Map`, the `json!` macro, or other serde_json-only APIs,
//! depend on `serde_json` directly — those are not part of the facade.

#[cfg(feature = "sonic")]
pub use sonic_rs::{from_slice, from_str, to_string, to_vec};

#[cfg(not(feature = "sonic"))]
pub use serde_json::{from_slice, from_str, to_string, to_vec};

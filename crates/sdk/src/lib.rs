//! Typed Rust client for the crabllm LLM API gateway.
//!
//! [`Client`] talks HTTP to a crabllm-compatible gateway (e.g. a deployed
//! `crabllm-proxy`) and implements [`crabllm_core::Provider`], so it drops into
//! any `Provider`-generic code. Retries and a per-attempt timeout are on by
//! default; configure them with [`Client::builder`]. The HTTP backend is
//! `crabllm-http` (hyper or reqwest, by feature), and streaming responses are
//! parsed by the shared [`crabllm_core::codec`] — the same code the gateway
//! uses, so client and server never drift.
//!
//! ```no_run
//! use crabllm_sdk::{Client, core::Provider};
//!
//! # async fn run(request: crabllm_sdk::core::anthropic::Request) -> Result<(), crabllm_sdk::core::Error> {
//! let client = Client::new("https://gateway.example.com", "sk-...");
//! let resp = client.anthropic_messages(&request).await?;
//! # let _ = resp; Ok(())
//! # }
//! ```
mod client;
mod provider;

pub use client::{Auth, Client, ClientBuilder, Route, route};

/// Re-export of `crabllm-core` so consumers get the wire types, the
/// [`Provider`](crabllm_core::Provider) trait, and [`Retrying`](crabllm_core::Retrying)
/// without a second dependency.
pub use crabllm_core as core;

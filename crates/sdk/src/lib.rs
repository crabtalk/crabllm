//! Typed Rust client for the crabllm LLM API gateway.
//!
//! [`Client`] talks HTTP to a crabllm-compatible gateway (e.g. a deployed
//! `crabllm-proxy`) and implements [`crabllm_core::Provider`], so it composes
//! with [`crabllm_core::Retrying`] and drops into any `Provider`-generic code.
//! Streaming responses are parsed by the shared [`crabllm_core::codec`], the
//! same code the gateway itself uses — so the client and server never drift.
//!
//! ```no_run
//! # async fn run() -> Result<(), crabllm_core::Error> {
//! use crabllm_sdk::{Client, core::{Provider, Retrying}};
//!
//! let client = Retrying::new(Client::new("https://gateway.example.com", "sk-..."));
//! let resp = client.anthropic_messages(&request).await?;
//! # let _ = resp; Ok(())
//! # }
//! ```
mod client;
mod http;
mod provider;

pub use client::Client;

/// Re-export of `crabllm-core` so consumers get the wire types, the
/// [`Provider`](crabllm_core::Provider) trait, and [`Retrying`](crabllm_core::Retrying)
/// without a second dependency.
pub use crabllm_core as core;

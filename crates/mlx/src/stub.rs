//! Non-Apple stub. Every call returns `Error::not_implemented` so the
//! crate compiles on Linux / Windows / etc. and downstream code that
//! references `crabllm_mlx::*` behind a `cfg` still type-checks.
//!
//! The public surface must stay in lockstep with the real `session.rs`,
//! `model.rs`, `pool.rs`, and `provider.rs`. Adding a public API in
//! one of those files without mirroring it here breaks the Linux CI
//! build.

use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Error, Provider,
};
use std::{path::Path, sync::Arc, sync::atomic::AtomicU32, time::Duration};

const STUB_MSG: &str = "mlx: only macOS and iOS (Apple Silicon) are supported";

// ---------- Session-level types ----------

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateOptions {
    pub seed: u64,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

pub struct GenerateRequest<'a> {
    pub messages_json: &'a str,
    pub tools_json: Option<&'a str>,
    pub options: GenerateOptions,
    pub cancel_flag: Option<&'a AtomicU32>,
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub text: String,
    pub tool_calls_json: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct StreamOutput {
    pub tool_calls_json: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct Session;

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn new(_model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub fn generate(&self, _req: &GenerateRequest<'_>) -> Result<GenerateOutput, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub fn generate_stream<F>(
        &self,
        _req: &GenerateRequest<'_>,
        _on_token: F,
    ) -> Result<StreamOutput, Error>
    where
        F: FnMut(&str) -> bool,
    {
        Err(Error::not_implemented(STUB_MSG))
    }
}

// ---------- High-level types (MlxModel / MlxPool / MlxProvider) ----------

#[derive(Clone)]
pub struct MlxModel;

impl MlxModel {
    pub async fn new(_model_id: impl Into<String>) -> Result<Self, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub fn name(&self) -> &str {
        ""
    }
}

impl Provider for MlxModel {
    async fn chat_completion(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    async fn chat_completion_stream(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }
}

#[derive(Debug, Default)]
pub struct MlxPool;

impl MlxPool {
    pub fn new() -> Self {
        Self
    }

    pub fn with_idle_timeout(self, _timeout: Duration) -> Self {
        self
    }

    pub async fn ensure_loaded(&self, _model_id: &str) -> Result<MlxModel, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub async fn evict(&self, _model_id: &str) {}

    pub async fn stop_all(&self) {}
}

#[derive(Clone, Debug)]
pub struct MlxProvider {
    _pool: Arc<MlxPool>,
}

impl MlxProvider {
    pub fn new(pool: Arc<MlxPool>) -> Self {
        Self { _pool: pool }
    }
}

impl Provider for MlxProvider {
    async fn chat_completion(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    async fn chat_completion_stream(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }
}

//! Non-Apple stub. Every call returns `Error::not_implemented`.

use crabllm_core::{
    AnthropicStreamEvent, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, Error, GeminiRequest, GeminiResponse, Provider,
};
use std::sync::Arc;

const STUB_MSG: &str = "mlx: only macOS and iOS (Apple Silicon) are supported";

#[derive(Debug)]
pub struct MlxPool;
unsafe impl Send for MlxPool {}
unsafe impl Sync for MlxPool {}

impl MlxPool {
    pub fn new(_idle_timeout_secs: u64) -> Result<Self, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }
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

    async fn anthropic_messages(
        &self,
        _request: &crabllm_core::AnthropicRequest,
    ) -> Result<crabllm_core::AnthropicResponse, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    async fn anthropic_messages_stream(
        &self,
        _request: &crabllm_core::AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    async fn gemini_generate_content_stream(
        &self,
        _model: &str,
        _request: &GeminiRequest,
    ) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }
}

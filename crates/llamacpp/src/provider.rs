use crate::pool::ServerPool;
use crabllm_core::{
    AnthropicStreamEvent, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, Error, Provider,
};
use crabllm_provider::openai_client;
use futures::StreamExt;
use std::sync::Arc;

/// Provider implementation backed by an on-demand `ServerPool` of
/// `llama-server` child processes.
///
/// Each trait call resolves the request's model name via
/// `ServerPool::ensure_running`, which spawns a new `llama-server`
/// subprocess on first reference and caches it until the idle timeout.
/// The returned base URL is an OpenAI-compatible `http://127.0.0.1:{port}/v1`
/// endpoint; forwarding goes through `crabllm-provider`'s shared
/// OpenAI HTTP helpers so the SSE parser and error shape stay identical
/// to the remote OpenAI path.
///
/// Cloning is cheap: `Arc<ServerPool>` + `reqwest::Client` (internally
/// `Arc<ClientRef>`). Clones share the same pool and connection state,
/// so multiple `Deployment` entries for different models can hold
/// independent clones without duplicating the backend.
#[derive(Clone)]
pub struct LlamaCppProvider {
    pool: Arc<ServerPool>,
    client: crabllm_provider::HttpClient,
}

impl LlamaCppProvider {
    pub fn new(pool: Arc<ServerPool>, client: crabllm_provider::HttpClient) -> Self {
        Self { pool, client }
    }
}

impl Provider for LlamaCppProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let base_url = self.pool.ensure_running(&request.model).await?;
        // llama-server ignores the bearer header — any empty string works.
        openai_client::chat_completion(&self.client, &base_url, "", request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let base_url = self.pool.ensure_running(&request.model).await?;
        let s = openai_client::chat_completion_stream(&self.client, &base_url, "", request).await?;
        Ok(s.boxed())
    }

    async fn anthropic_messages(
        &self,
        request: &crabllm_core::AnthropicRequest,
    ) -> Result<crabllm_core::AnthropicResponse, Error> {
        let chat_req = ChatCompletionRequest::from(request.clone());
        let resp = self.chat_completion(&chat_req).await?;
        crabllm_core::AnthropicResponse::try_from(resp)
    }

    async fn anthropic_messages_stream(
        &self,
        request: &crabllm_core::AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        crabllm_provider::anthropic_stream_via_chat(self, request).await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &crabllm_core::GeminiRequest,
    ) -> Result<BoxStream<'static, Result<crabllm_core::GeminiResponse, Error>>, Error> {
        crabllm_provider::gemini_stream_via_chat(self, model, request).await
    }
}

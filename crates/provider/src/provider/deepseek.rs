use crate::provider::{anthropic::anthropic_event_stream, openai};
use crate::{ByteStream, HttpClient};
use bytes::Bytes;
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error,
    Provider,
};
use futures::stream::StreamExt;

pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Debug, Clone)]
pub struct DeepseekProvider {
    pub(crate) client: HttpClient,
    pub(crate) openai_base_url: String,
    pub(crate) anthropic_base_url: String,
    pub(crate) api_key: String,
}

impl Provider for DeepseekProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        openai::chat_completion(&self.client, &self.openai_base_url, &self.api_key, request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let s = openai::chat_completion_stream(
            &self.client,
            &self.openai_base_url,
            &self.api_key,
            request,
        )
        .await?;
        Ok(s.boxed())
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        openai::embedding(&self.client, &self.openai_base_url, &self.api_key, request).await
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let body =
            crabllm_core::json::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
        let resp_bytes = anthropic_messages_raw(
            &self.client,
            &self.anthropic_base_url,
            &self.api_key,
            body.into(),
        )
        .await?;
        crabllm_core::json::from_slice(&resp_bytes).map_err(|e| Error::Internal(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Internal(e.to_string()))?;
        let byte_stream = anthropic_messages_stream(
            &self.client,
            &self.anthropic_base_url,
            &self.api_key,
            body.into(),
        )
        .await?;
        Ok(anthropic_event_stream(byte_stream, request.model.clone()).boxed())
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &crabllm_core::GeminiRequest,
    ) -> Result<BoxStream<'static, Result<crabllm_core::GeminiResponse, Error>>, Error> {
        crate::gemini_stream_via_chat(self, model, request).await
    }

    fn is_openai_compat(&self) -> bool {
        true
    }

    fn is_anthropic_compat(&self) -> bool {
        true
    }

    async fn chat_completion_raw(&self, _model: &str, raw_body: Bytes) -> Result<Bytes, Error> {
        openai::chat_completion_raw(&self.client, &self.openai_base_url, &self.api_key, raw_body)
            .await
    }

    async fn anthropic_messages_raw(&self, raw_body: Bytes) -> Result<Bytes, Error> {
        anthropic_messages_raw(
            &self.client,
            &self.anthropic_base_url,
            &self.api_key,
            raw_body,
        )
        .await
    }

    async fn anthropic_messages_stream_raw(
        &self,
        raw_body: Bytes,
    ) -> Result<crabllm_core::ByteStream, Error> {
        anthropic_messages_stream(
            &self.client,
            &self.anthropic_base_url,
            &self.api_key,
            raw_body,
        )
        .await
    }
}

/// Forward raw Anthropic-format JSON bytes to DeepSeek's Anthropic-
/// compatible endpoint. Uses `Authorization: Bearer` (not `x-api-key`)
/// because DeepSeek unifies auth across both endpoints.
pub async fn anthropic_messages_raw(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<Bytes, Error> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let bearer = format!("Bearer {api_key}");
    let headers = [
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
        ("authorization", bearer.as_str()),
    ];
    let resp = client
        .post(&url, &headers, raw_body)
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
        });
    }

    Ok(resp.body)
}

/// Stream raw Anthropic SSE bytes from DeepSeek's Anthropic-compatible endpoint.
pub async fn anthropic_messages_stream(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<ByteStream, Error> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let bearer = format!("Bearer {api_key}");
    let headers = [
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
        ("authorization", bearer.as_str()),
    ];
    client.post_stream(&url, &headers, raw_body).await
}

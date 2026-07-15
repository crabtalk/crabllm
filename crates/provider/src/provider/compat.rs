use crate::provider::openai;
use crate::{ByteStream, HttpClient};
use bytes::Bytes;
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error,
    Provider, codec::anthropic::anthropic_event_stream,
};
use futures::stream::StreamExt;

/// A built-in provider that speaks OpenAI-compat + native Anthropic. Every
/// such provider differs only by these three strings, so they are data, not
/// code — add a row to [`COMPAT`] to add a provider. A provider that needs
/// genuinely different *behavior* graduates out of this table into its own
/// module; none do today.
pub struct CompatSpec {
    /// The `kind` string that selects this provider (also the natural map key).
    pub name: &'static str,
    /// OpenAI-compatible base — `/chat/completions` is appended.
    pub openai_base_url: &'static str,
    /// Anthropic-compatible base — `/messages` is appended, so include `/v1`.
    pub anthropic_base_url: &'static str,
}

pub const COMPAT: &[CompatSpec] = &[
    CompatSpec {
        name: "deepseek",
        openai_base_url: "https://api.deepseek.com/v1",
        anthropic_base_url: "https://api.deepseek.com/anthropic",
    },
    CompatSpec {
        name: "zai",
        openai_base_url: "https://api.z.ai/api/paas/v4",
        anthropic_base_url: "https://api.z.ai/api/anthropic/v1",
    },
    CompatSpec {
        name: "qwen",
        openai_base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        anthropic_base_url: "https://dashscope-intl.aliyuncs.com/apps/anthropic/v1",
    },
];

/// Resolve a `kind`/name string to its compat spec, if it names one.
pub fn lookup(name: &str) -> Option<&'static CompatSpec> {
    COMPAT.iter().find(|s| s.name == name)
}

#[derive(Debug, Clone)]
pub struct CompatProvider {
    pub(crate) client: HttpClient,
    pub(crate) openai_base_url: String,
    pub(crate) anthropic_base_url: String,
    pub(crate) api_key: String,
}

impl Provider for CompatProvider {
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
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let resp_bytes = anthropic_messages_raw(
            &self.client,
            &self.anthropic_base_url,
            &self.api_key,
            body.into(),
        )
        .await?;
        crabllm_core::json::from_slice(&resp_bytes).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Encode(e.to_string()))?;
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

/// Forward raw Anthropic-format JSON bytes to a compat provider's
/// Anthropic-compatible endpoint. Uses `Authorization: Bearer` — every compat
/// provider accepts it (some also accept `x-api-key`).
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
    let resp = client.post(&url, &headers, raw_body).await?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
            retry_after: resp.retry_after,
        });
    }

    Ok(resp.body)
}

/// Stream raw Anthropic SSE bytes from a compat provider's Anthropic endpoint.
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

use bytes::Bytes;
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, AudioSpeechRequest, BoxStream,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, Error, GeminiRequest, GeminiResponse, ImageRequest, MultipartField,
    Provider, ProviderConfig, ProviderKind, ir,
};
use futures::StreamExt;
pub use registry::{Deployment, ProviderRegistry};

mod client;
mod provider;
mod registry;

pub use client::{ByteStream, HttpClient};
pub use crabllm_core::codec::anthropic::{
    anthropic_event_stream, anthropic_events_to_chunks, chunks_to_anthropic_events,
};
pub use crabllm_core::codec::gemini::{
    chunks_to_gemini_responses, gemini_event_stream, gemini_responses_to_chunks,
};
pub use provider::schema;
pub use provider::{
    anthropic::AnthropicProvider, azure::AzureProvider, deepseek::DeepseekProvider,
    google::GoogleProvider, openai::OpenaiProvider,
};

#[cfg(feature = "bedrock")]
pub use provider::bedrock::BedrockProvider;

#[cfg(not(feature = "bedrock"))]
mod bedrock_stub {
    #[derive(Debug, Clone)]
    pub struct BedrockProvider;

    impl crabllm_core::Provider for BedrockProvider {
        async fn chat_completion(
            &self,
            _request: &crabllm_core::ChatCompletionRequest,
        ) -> Result<crabllm_core::ChatCompletionResponse, crabllm_core::Error> {
            Err(crabllm_core::Error::not_implemented("bedrock chat"))
        }

        async fn chat_completion_stream(
            &self,
            _request: &crabllm_core::ChatCompletionRequest,
        ) -> Result<
            crabllm_core::BoxStream<
                'static,
                Result<crabllm_core::ChatCompletionChunk, crabllm_core::Error>,
            >,
            crabllm_core::Error,
        > {
            Err(crabllm_core::Error::not_implemented("bedrock streaming"))
        }

        async fn anthropic_messages(
            &self,
            _request: &crabllm_core::AnthropicRequest,
        ) -> Result<crabllm_core::AnthropicResponse, crabllm_core::Error> {
            Err(crabllm_core::Error::not_implemented("bedrock anthropic"))
        }

        async fn anthropic_messages_stream(
            &self,
            _request: &crabllm_core::AnthropicRequest,
        ) -> Result<
            crabllm_core::BoxStream<
                'static,
                Result<crabllm_core::AnthropicStreamEvent, crabllm_core::Error>,
            >,
            crabllm_core::Error,
        > {
            Err(crabllm_core::Error::not_implemented(
                "bedrock anthropic streaming",
            ))
        }

        async fn gemini_generate_content_stream(
            &self,
            _model: &str,
            _request: &crabllm_core::GeminiRequest,
        ) -> Result<
            crabllm_core::BoxStream<
                'static,
                Result<crabllm_core::GeminiResponse, crabllm_core::Error>,
            >,
            crabllm_core::Error,
        > {
            Err(crabllm_core::Error::not_implemented(
                "bedrock gemini streaming",
            ))
        }
    }
}

#[cfg(not(feature = "bedrock"))]
pub use bedrock_stub::BedrockProvider;

/// Exposed so `crabllm-llamacpp` can reuse the OpenAI-compatible HTTP
/// helpers against the child llama-server process. Append-only surface —
/// add a re-export here only when a local backend actually needs the
/// function.
pub mod openai_client {
    pub use crate::provider::openai::{chat_completion, chat_completion_stream, embedding};
}

/// Shared fallback for providers that don't natively speak Anthropic
/// streaming: convert the request → `chat_completion_stream` → wrap
/// each chunk as an `AnthropicStreamEvent`.
pub async fn anthropic_stream_via_chat(
    provider: &(impl Provider + ?Sized),
    request: &AnthropicRequest,
) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
    let mut ir_req = ir::Request::from(request.clone());
    ir_req.stream = true;
    let chat_req = ChatCompletionRequest::from(&ir_req);
    let chunks = provider.chat_completion_stream(&chat_req).await?;
    Ok(chunks_to_anthropic_events(chunks).boxed())
}

/// Shared fallback for providers that don't natively speak Gemini
/// streaming: convert the request → `chat_completion_stream` → wrap
/// each chunk as a `GeminiResponse`.
pub async fn gemini_stream_via_chat(
    provider: &(impl Provider + ?Sized),
    model: &str,
    request: &GeminiRequest,
) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
    let mut ir_req = ir::Request::from(request);
    ir_req.model = model.to_string();
    ir_req.stream = true;
    let chat_req = ChatCompletionRequest::from(&ir_req);
    let chunks = provider.chat_completion_stream(&chat_req).await?;
    Ok(chunks_to_gemini_responses(chunks).boxed())
}

/// A configured remote-API provider, ready to dispatch requests.
///
/// Each variant wraps a provider struct that implements `Provider`
/// directly. `RemoteProvider` is a thin delegating enum — the real
/// implementation lives in the per-provider structs. Cloning is cheap
/// because `HttpClient` is internally `Arc`-shared.
#[derive(Debug, Clone)]
pub enum RemoteProvider {
    Openai(OpenaiProvider),
    Anthropic(AnthropicProvider),
    Deepseek(DeepseekProvider),
    Google(GoogleProvider),
    Bedrock(BedrockProvider),
    Azure(AzureProvider),
}

/// Build the shared [`HttpClient`] used by every `RemoteProvider`.
/// Called once at registry construction and cloned into every provider,
/// so all share a single connection pool, DNS resolver, and TLS state.
pub fn make_client() -> HttpClient {
    HttpClient::new()
}

/// Strip known endpoint suffixes so users can paste either a bare origin
/// (`https://api.openai.com/v1`) or a full endpoint URL
/// (`https://api.openai.com/v1/chat/completions`) and get the same result.
///
/// Only the OpenAI-shaped endpoints are stripped: `/chat/completions`,
/// `/embeddings`, `/audio/transcriptions`, `/audio/speech`,
/// `/images/generations`. Anthropic appends `/messages` itself, so
/// stripping is not needed there.
fn normalize_base_url(url: &str) -> String {
    let url = url.trim_end_matches('/');
    for suffix in [
        "/chat/completions",
        "/embeddings",
        "/audio/transcriptions",
        "/audio/speech",
        "/images/generations",
    ] {
        if let Some(stripped) = url.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    url.to_string()
}

impl RemoteProvider {
    pub fn new(config: &ProviderConfig, client: HttpClient) -> Self {
        let api_key = config.api_key.clone().unwrap_or_default();
        match config.effective_kind() {
            ProviderKind::Openai => RemoteProvider::Openai(OpenaiProvider {
                client,
                base_url: normalize_base_url(
                    &config
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                ),
                api_key,
            }),
            ProviderKind::Anthropic => RemoteProvider::Anthropic(AnthropicProvider {
                client,
                base_url: config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| provider::anthropic::DEFAULT_BASE_URL.to_string()),
                api_key,
            }),
            ProviderKind::Deepseek => {
                let base = config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| provider::deepseek::DEFAULT_BASE_URL.to_string());
                let base = base.trim_end_matches('/');
                RemoteProvider::Deepseek(DeepseekProvider {
                    client,
                    openai_base_url: normalize_base_url(&format!("{base}/v1")),
                    anthropic_base_url: format!("{base}/anthropic"),
                    api_key,
                })
            }
            ProviderKind::Google => RemoteProvider::Google(GoogleProvider { client, api_key }),
            ProviderKind::Ollama => RemoteProvider::Openai(OpenaiProvider {
                client,
                base_url: normalize_base_url(
                    &config
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "http://localhost:11434/v1".to_string()),
                ),
                api_key,
            }),
            ProviderKind::Azure => RemoteProvider::Azure(AzureProvider {
                client,
                base_url: normalize_base_url(&config.base_url.clone().unwrap_or_default()),
                api_key,
                api_version: config
                    .api_version
                    .clone()
                    .unwrap_or_else(|| "2024-02-15-preview".to_string()),
            }),
            ProviderKind::Bedrock => {
                #[cfg(feature = "bedrock")]
                {
                    RemoteProvider::Bedrock(BedrockProvider {
                        client,
                        region: config.region.clone().unwrap_or_default(),
                        access_key: config.access_key.clone().unwrap_or_default(),
                        secret_key: config.secret_key.clone().unwrap_or_default(),
                    })
                }
                #[cfg(not(feature = "bedrock"))]
                {
                    let _ = client;
                    RemoteProvider::Bedrock(BedrockProvider)
                }
            }
            ProviderKind::Custom(_) => RemoteProvider::Openai(OpenaiProvider {
                client,
                base_url: normalize_base_url(&config.base_url.clone().unwrap_or_default()),
                api_key,
            }),
        }
    }
}

pub(crate) fn rebuild_multipart(fields: &[MultipartField]) -> (Bytes, String) {
    let boundary = format!("crabllm-{:016x}", rand::random::<u64>());
    let mut buf = Vec::new();
    for field in fields {
        buf.extend_from_slice(b"--");
        buf.extend_from_slice(boundary.as_bytes());
        buf.extend_from_slice(b"\r\n");
        buf.extend_from_slice(b"Content-Disposition: form-data; name=\"");
        buf.extend_from_slice(field.name.as_bytes());
        buf.push(b'"');
        if let Some(ref filename) = field.filename {
            buf.extend_from_slice(b"; filename=\"");
            buf.extend_from_slice(filename.as_bytes());
            buf.push(b'"');
        }
        buf.extend_from_slice(b"\r\n");
        if let Some(ref ct) = field.content_type {
            buf.extend_from_slice(b"Content-Type: ");
            buf.extend_from_slice(ct.as_bytes());
            buf.extend_from_slice(b"\r\n");
        }
        buf.extend_from_slice(b"\r\n");
        buf.extend_from_slice(&field.bytes);
        buf.extend_from_slice(b"\r\n");
    }
    buf.extend_from_slice(b"--");
    buf.extend_from_slice(boundary.as_bytes());
    buf.extend_from_slice(b"--\r\n");
    (Bytes::from(buf), boundary)
}

impl Provider for RemoteProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        match self {
            Self::Openai(p) => p.chat_completion(request).await,
            Self::Anthropic(p) => p.chat_completion(request).await,
            Self::Deepseek(p) => p.chat_completion(request).await,
            Self::Google(p) => p.chat_completion(request).await,
            Self::Bedrock(p) => p.chat_completion(request).await,
            Self::Azure(p) => p.chat_completion(request).await,
        }
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        match self {
            Self::Openai(p) => p.chat_completion_stream(request).await,
            Self::Anthropic(p) => p.chat_completion_stream(request).await,
            Self::Deepseek(p) => p.chat_completion_stream(request).await,
            Self::Google(p) => p.chat_completion_stream(request).await,
            Self::Bedrock(p) => p.chat_completion_stream(request).await,
            Self::Azure(p) => p.chat_completion_stream(request).await,
        }
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        match self {
            Self::Openai(p) => p.anthropic_messages(request).await,
            Self::Anthropic(p) => p.anthropic_messages(request).await,
            Self::Deepseek(p) => p.anthropic_messages(request).await,
            Self::Google(p) => p.anthropic_messages(request).await,
            Self::Bedrock(p) => p.anthropic_messages(request).await,
            Self::Azure(p) => p.anthropic_messages(request).await,
        }
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        match self {
            Self::Openai(p) => p.anthropic_messages_stream(request).await,
            Self::Anthropic(p) => p.anthropic_messages_stream(request).await,
            Self::Deepseek(p) => p.anthropic_messages_stream(request).await,
            Self::Google(p) => p.anthropic_messages_stream(request).await,
            Self::Bedrock(p) => p.anthropic_messages_stream(request).await,
            Self::Azure(p) => p.anthropic_messages_stream(request).await,
        }
    }

    async fn complete(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<crabllm_core::ir::Response, Error> {
        match self {
            Self::Openai(p) => p.complete(request).await,
            Self::Anthropic(p) => p.complete(request).await,
            Self::Deepseek(p) => p.complete(request).await,
            Self::Google(p) => p.complete(request).await,
            Self::Bedrock(p) => p.complete(request).await,
            Self::Azure(p) => p.complete(request).await,
        }
    }

    async fn complete_stream(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ir::StreamEvent, Error>>, Error> {
        match self {
            Self::Openai(p) => p.complete_stream(request).await,
            Self::Anthropic(p) => p.complete_stream(request).await,
            Self::Deepseek(p) => p.complete_stream(request).await,
            Self::Google(p) => p.complete_stream(request).await,
            Self::Bedrock(p) => p.complete_stream(request).await,
            Self::Azure(p) => p.complete_stream(request).await,
        }
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        match self {
            Self::Openai(p) => p.embedding(request).await,
            Self::Anthropic(p) => p.embedding(request).await,
            Self::Deepseek(p) => p.embedding(request).await,
            Self::Google(p) => p.embedding(request).await,
            Self::Bedrock(p) => p.embedding(request).await,
            Self::Azure(p) => p.embedding(request).await,
        }
    }

    async fn image_generation(&self, request: &ImageRequest) -> Result<(Bytes, String), Error> {
        match self {
            Self::Openai(p) => p.image_generation(request).await,
            Self::Anthropic(p) => p.image_generation(request).await,
            Self::Deepseek(p) => p.image_generation(request).await,
            Self::Google(p) => p.image_generation(request).await,
            Self::Bedrock(p) => p.image_generation(request).await,
            Self::Azure(p) => p.image_generation(request).await,
        }
    }

    async fn audio_speech(&self, request: &AudioSpeechRequest) -> Result<(Bytes, String), Error> {
        match self {
            Self::Openai(p) => p.audio_speech(request).await,
            Self::Anthropic(p) => p.audio_speech(request).await,
            Self::Deepseek(p) => p.audio_speech(request).await,
            Self::Google(p) => p.audio_speech(request).await,
            Self::Bedrock(p) => p.audio_speech(request).await,
            Self::Azure(p) => p.audio_speech(request).await,
        }
    }

    async fn audio_transcription(
        &self,
        model: &str,
        fields: &[MultipartField],
    ) -> Result<(Bytes, String), Error> {
        match self {
            Self::Openai(p) => p.audio_transcription(model, fields).await,
            Self::Anthropic(p) => p.audio_transcription(model, fields).await,
            Self::Deepseek(p) => p.audio_transcription(model, fields).await,
            Self::Google(p) => p.audio_transcription(model, fields).await,
            Self::Bedrock(p) => p.audio_transcription(model, fields).await,
            Self::Azure(p) => p.audio_transcription(model, fields).await,
        }
    }

    fn is_openai_compat(&self) -> bool {
        match self {
            Self::Openai(p) => p.is_openai_compat(),
            Self::Anthropic(p) => p.is_openai_compat(),
            Self::Deepseek(p) => p.is_openai_compat(),
            Self::Google(p) => p.is_openai_compat(),
            Self::Bedrock(p) => p.is_openai_compat(),
            Self::Azure(p) => p.is_openai_compat(),
        }
    }

    fn is_anthropic_compat(&self) -> bool {
        match self {
            Self::Openai(p) => p.is_anthropic_compat(),
            Self::Anthropic(p) => p.is_anthropic_compat(),
            Self::Deepseek(p) => p.is_anthropic_compat(),
            Self::Google(p) => p.is_anthropic_compat(),
            Self::Bedrock(p) => p.is_anthropic_compat(),
            Self::Azure(p) => p.is_anthropic_compat(),
        }
    }

    async fn chat_completion_stream_passthrough(
        &self,
        model: &str,
        body_stream: crabllm_core::ByteStream,
    ) -> Result<crabllm_core::ByteStream, Error> {
        match self {
            Self::Openai(p) => {
                p.chat_completion_stream_passthrough(model, body_stream)
                    .await
            }
            Self::Anthropic(p) => {
                p.chat_completion_stream_passthrough(model, body_stream)
                    .await
            }
            Self::Deepseek(p) => {
                p.chat_completion_stream_passthrough(model, body_stream)
                    .await
            }
            Self::Google(p) => {
                p.chat_completion_stream_passthrough(model, body_stream)
                    .await
            }
            Self::Bedrock(p) => {
                p.chat_completion_stream_passthrough(model, body_stream)
                    .await
            }
            Self::Azure(p) => {
                p.chat_completion_stream_passthrough(model, body_stream)
                    .await
            }
        }
    }

    async fn chat_completion_stream_raw(
        &self,
        model: &str,
        raw_body: Bytes,
    ) -> Result<crabllm_core::ByteStream, Error> {
        match self {
            Self::Openai(p) => p.chat_completion_stream_raw(model, raw_body).await,
            Self::Anthropic(p) => p.chat_completion_stream_raw(model, raw_body).await,
            Self::Deepseek(p) => p.chat_completion_stream_raw(model, raw_body).await,
            Self::Google(p) => p.chat_completion_stream_raw(model, raw_body).await,
            Self::Bedrock(p) => p.chat_completion_stream_raw(model, raw_body).await,
            Self::Azure(p) => p.chat_completion_stream_raw(model, raw_body).await,
        }
    }

    async fn chat_completion_raw(&self, model: &str, raw_body: Bytes) -> Result<Bytes, Error> {
        match self {
            Self::Openai(p) => p.chat_completion_raw(model, raw_body).await,
            Self::Anthropic(p) => p.chat_completion_raw(model, raw_body).await,
            Self::Deepseek(p) => p.chat_completion_raw(model, raw_body).await,
            Self::Google(p) => p.chat_completion_raw(model, raw_body).await,
            Self::Bedrock(p) => p.chat_completion_raw(model, raw_body).await,
            Self::Azure(p) => p.chat_completion_raw(model, raw_body).await,
        }
    }

    async fn anthropic_messages_raw(&self, raw_body: Bytes) -> Result<Bytes, Error> {
        match self {
            Self::Openai(p) => p.anthropic_messages_raw(raw_body).await,
            Self::Anthropic(p) => p.anthropic_messages_raw(raw_body).await,
            Self::Deepseek(p) => p.anthropic_messages_raw(raw_body).await,
            Self::Google(p) => p.anthropic_messages_raw(raw_body).await,
            Self::Bedrock(p) => p.anthropic_messages_raw(raw_body).await,
            Self::Azure(p) => p.anthropic_messages_raw(raw_body).await,
        }
    }

    async fn anthropic_messages_stream_raw(&self, raw_body: Bytes) -> Result<ByteStream, Error> {
        match self {
            Self::Openai(p) => p.anthropic_messages_stream_raw(raw_body).await,
            Self::Anthropic(p) => p.anthropic_messages_stream_raw(raw_body).await,
            Self::Deepseek(p) => p.anthropic_messages_stream_raw(raw_body).await,
            Self::Google(p) => p.anthropic_messages_stream_raw(raw_body).await,
            Self::Bedrock(p) => p.anthropic_messages_stream_raw(raw_body).await,
            Self::Azure(p) => p.anthropic_messages_stream_raw(raw_body).await,
        }
    }

    fn is_gemini_compat(&self) -> bool {
        match self {
            Self::Openai(p) => p.is_gemini_compat(),
            Self::Anthropic(p) => p.is_gemini_compat(),
            Self::Deepseek(p) => p.is_gemini_compat(),
            Self::Google(p) => p.is_gemini_compat(),
            Self::Bedrock(p) => p.is_gemini_compat(),
            Self::Azure(p) => p.is_gemini_compat(),
        }
    }

    async fn gemini_generate_content(
        &self,
        model: &str,
        request: &crabllm_core::GeminiRequest,
    ) -> Result<crabllm_core::GeminiResponse, Error> {
        match self {
            Self::Openai(p) => p.gemini_generate_content(model, request).await,
            Self::Anthropic(p) => p.gemini_generate_content(model, request).await,
            Self::Deepseek(p) => p.gemini_generate_content(model, request).await,
            Self::Google(p) => p.gemini_generate_content(model, request).await,
            Self::Bedrock(p) => p.gemini_generate_content(model, request).await,
            Self::Azure(p) => p.gemini_generate_content(model, request).await,
        }
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &crabllm_core::GeminiRequest,
    ) -> Result<BoxStream<'static, Result<crabllm_core::GeminiResponse, Error>>, Error> {
        match self {
            Self::Openai(p) => p.gemini_generate_content_stream(model, request).await,
            Self::Anthropic(p) => p.gemini_generate_content_stream(model, request).await,
            Self::Deepseek(p) => p.gemini_generate_content_stream(model, request).await,
            Self::Google(p) => p.gemini_generate_content_stream(model, request).await,
            Self::Bedrock(p) => p.gemini_generate_content_stream(model, request).await,
            Self::Azure(p) => p.gemini_generate_content_stream(model, request).await,
        }
    }

    async fn gemini_generate_content_raw(
        &self,
        model: &str,
        raw_body: Bytes,
    ) -> Result<Bytes, Error> {
        match self {
            Self::Openai(p) => p.gemini_generate_content_raw(model, raw_body).await,
            Self::Anthropic(p) => p.gemini_generate_content_raw(model, raw_body).await,
            Self::Deepseek(p) => p.gemini_generate_content_raw(model, raw_body).await,
            Self::Google(p) => p.gemini_generate_content_raw(model, raw_body).await,
            Self::Bedrock(p) => p.gemini_generate_content_raw(model, raw_body).await,
            Self::Azure(p) => p.gemini_generate_content_raw(model, raw_body).await,
        }
    }

    async fn gemini_generate_content_stream_raw(
        &self,
        model: &str,
        raw_body: Bytes,
    ) -> Result<crabllm_core::ByteStream, Error> {
        match self {
            Self::Openai(p) => p.gemini_generate_content_stream_raw(model, raw_body).await,
            Self::Anthropic(p) => p.gemini_generate_content_stream_raw(model, raw_body).await,
            Self::Deepseek(p) => p.gemini_generate_content_stream_raw(model, raw_body).await,
            Self::Google(p) => p.gemini_generate_content_stream_raw(model, raw_body).await,
            Self::Bedrock(p) => p.gemini_generate_content_stream_raw(model, raw_body).await,
            Self::Azure(p) => p.gemini_generate_content_stream_raw(model, raw_body).await,
        }
    }
}

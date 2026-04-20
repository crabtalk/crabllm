#[cfg(all(feature = "rustls", feature = "native-tls"))]
compile_error!("crabllm-provider: features `rustls` and `native-tls` are mutually exclusive");

#[cfg(not(any(feature = "rustls", feature = "native-tls")))]
compile_error!("crabllm-provider: enable exactly one of `rustls` or `native-tls`");

use bytes::Bytes;
use crabllm_core::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, ImageRequest,
    MultipartField, Provider, ProviderConfig, ProviderKind,
};
use futures::stream::StreamExt;
pub use registry::{Deployment, ProviderRegistry};

mod client;
mod provider;
mod registry;

pub use client::{ByteStream, HttpClient};
pub use provider::schema;

/// Exposed so `crabllm-llamacpp` can reuse the OpenAI-compatible HTTP
/// helpers against the child llama-server process. Append-only surface —
/// add a re-export here only when a local backend actually needs the
/// function.
pub mod openai_client {
    pub use crate::provider::openai::{chat_completion, chat_completion_stream, embedding};
}

/// A configured remote-API provider, ready to dispatch requests.
///
/// Each variant carries a `HttpClient` so the provider trait
/// implementation needs no shared client passed by the caller. Cloning a
/// `RemoteProvider` is cheap — `HttpClient` is internally `Arc`-shared.
#[derive(Debug, Clone)]
pub enum RemoteProvider {
    /// OpenAI-compatible providers (OpenAI, Ollama, vLLM, Groq, etc.).
    /// Request body is forwarded as-is with URL + auth rewrite.
    Openai {
        client: HttpClient,
        base_url: String,
        api_key: String,
    },
    /// Anthropic Messages API. Requires request/response translation.
    Anthropic { client: HttpClient, api_key: String },
    /// Google Gemini API. Requires request/response translation.
    Google { client: HttpClient, api_key: String },
    /// AWS Bedrock. Requires SigV4 signing + translation.
    Bedrock {
        client: HttpClient,
        region: String,
        access_key: String,
        secret_key: String,
    },
    /// Azure OpenAI. Uses deployment-based URL and api-key header.
    Azure {
        client: HttpClient,
        base_url: String,
        api_key: String,
        api_version: String,
    },
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
/// `/images/generations`. Anthropic, Google, and Bedrock don't take a
/// `base_url` field at all, so this function is never called for them.
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
    /// Build a `RemoteProvider` from a `ProviderConfig`, reusing a shared
    /// `HttpClient`. Cloning the client is cheap — internally it's
    /// `Arc<ClientRef>` — so every provider returned by this constructor
    /// dispatches through the same connection pool.
    ///
    /// Routes via [`ProviderConfig::effective_kind`] so a config with
    /// `kind = "openai"` and a `base_url` containing "anthropic" auto-upgrades
    /// to the Anthropic dispatch path. Base URLs are normalized so a pasted
    /// full endpoint URL (e.g. `…/v1/chat/completions`) collapses to the bare
    /// origin (`…/v1`).
    pub fn new(config: &ProviderConfig, client: HttpClient) -> Self {
        match config.effective_kind() {
            ProviderKind::Openai => RemoteProvider::Openai {
                client,
                base_url: normalize_base_url(
                    &config
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                ),
                api_key: config.api_key.clone().unwrap_or_default(),
            },
            ProviderKind::Anthropic => RemoteProvider::Anthropic {
                client,
                api_key: config.api_key.clone().unwrap_or_default(),
            },
            ProviderKind::Google => RemoteProvider::Google {
                client,
                api_key: config.api_key.clone().unwrap_or_default(),
            },
            ProviderKind::Ollama => RemoteProvider::Openai {
                client,
                base_url: normalize_base_url(
                    &config
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "http://localhost:11434/v1".to_string()),
                ),
                api_key: config.api_key.clone().unwrap_or_default(),
            },
            ProviderKind::Azure => RemoteProvider::Azure {
                client,
                base_url: normalize_base_url(&config.base_url.clone().unwrap_or_default()),
                api_key: config.api_key.clone().unwrap_or_default(),
                api_version: config
                    .api_version
                    .clone()
                    .unwrap_or_else(|| "2024-02-15-preview".to_string()),
            },
            ProviderKind::Bedrock => RemoteProvider::Bedrock {
                client,
                region: config.region.clone().unwrap_or_default(),
                access_key: config.access_key.clone().unwrap_or_default(),
                secret_key: config.secret_key.clone().unwrap_or_default(),
            },
            // Self-defined kind — must be OpenAI-compatible; base_url is
            // required (enforced by ProviderConfig::validate).
            ProviderKind::Custom(_) => RemoteProvider::Openai {
                client,
                base_url: normalize_base_url(&config.base_url.clone().unwrap_or_default()),
                api_key: config.api_key.clone().unwrap_or_default(),
            },
        }
    }
}

/// Build raw multipart body bytes and boundary from the provider-trait-
/// friendly `MultipartField` representation. Returns `(body, boundary)`.
fn rebuild_multipart(fields: &[MultipartField]) -> (Bytes, String) {
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
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => provider::openai::chat_completion(client, base_url, api_key, request).await,
            RemoteProvider::Anthropic { client, api_key } => {
                provider::anthropic::chat_completion(client, api_key, request).await
            }
            RemoteProvider::Google { client, api_key } => {
                provider::google::chat_completion(client, api_key, request).await
            }
            #[cfg(feature = "provider-bedrock")]
            RemoteProvider::Bedrock {
                client,
                region,
                access_key,
                secret_key,
            } => {
                provider::bedrock::chat_completion(client, region, access_key, secret_key, request)
                    .await
            }
            #[cfg(not(feature = "provider-bedrock"))]
            RemoteProvider::Bedrock { .. } => Err(provider::bedrock::not_implemented("chat")),
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                provider::azure::chat_completion(client, base_url, api_key, api_version, request)
                    .await
            }
        }
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        match self {
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => {
                let s =
                    provider::openai::chat_completion_stream(client, base_url, api_key, request)
                        .await?;
                Ok(s.boxed())
            }
            RemoteProvider::Anthropic { client, api_key } => {
                let s = provider::anthropic::chat_completion_stream(
                    client,
                    api_key,
                    request,
                    &request.model,
                )
                .await?;
                Ok(s.boxed())
            }
            RemoteProvider::Google { client, api_key } => {
                let s = provider::google::chat_completion_stream(
                    client,
                    api_key,
                    request,
                    &request.model,
                )
                .await?;
                Ok(s.boxed())
            }
            #[cfg(feature = "provider-bedrock")]
            RemoteProvider::Bedrock {
                client,
                region,
                access_key,
                secret_key,
            } => {
                let s = provider::bedrock::chat_completion_stream(
                    client,
                    region,
                    access_key,
                    secret_key,
                    request,
                    &request.model,
                )
                .await?;
                Ok(s.boxed())
            }
            #[cfg(not(feature = "provider-bedrock"))]
            RemoteProvider::Bedrock { .. } => Err(provider::bedrock::not_implemented("streaming")),
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                let s = provider::azure::chat_completion_stream(
                    client,
                    base_url,
                    api_key,
                    api_version,
                    request,
                )
                .await?;
                Ok(s.boxed())
            }
        }
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        match self {
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => provider::openai::embedding(client, base_url, api_key, request).await,
            RemoteProvider::Anthropic { .. } => {
                Err(provider::anthropic::not_implemented("embedding"))
            }
            RemoteProvider::Google { .. } => Err(provider::google::not_implemented("embedding")),
            RemoteProvider::Bedrock { .. } => Err(provider::bedrock::not_implemented("embedding")),
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => provider::azure::embedding(client, base_url, api_key, api_version, request).await,
        }
    }

    async fn image_generation(&self, request: &ImageRequest) -> Result<(Bytes, String), Error> {
        match self {
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => provider::openai::image_generation(client, base_url, api_key, request).await,
            RemoteProvider::Anthropic { .. } => {
                Err(provider::anthropic::not_implemented("image_generation"))
            }
            RemoteProvider::Google { .. } => {
                Err(provider::google::not_implemented("image_generation"))
            }
            RemoteProvider::Bedrock { .. } => {
                Err(provider::bedrock::not_implemented("image_generation"))
            }
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                provider::azure::image_generation(client, base_url, api_key, api_version, request)
                    .await
            }
        }
    }

    async fn audio_speech(&self, request: &AudioSpeechRequest) -> Result<(Bytes, String), Error> {
        match self {
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => provider::openai::audio_speech(client, base_url, api_key, request).await,
            RemoteProvider::Anthropic { .. } => {
                Err(provider::anthropic::not_implemented("audio_speech"))
            }
            RemoteProvider::Google { .. } => Err(provider::google::not_implemented("audio_speech")),
            RemoteProvider::Bedrock { .. } => {
                Err(provider::bedrock::not_implemented("audio_speech"))
            }
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                provider::azure::audio_speech(client, base_url, api_key, api_version, request).await
            }
        }
    }

    async fn audio_transcription(
        &self,
        model: &str,
        fields: &[MultipartField],
    ) -> Result<(Bytes, String), Error> {
        match self {
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => {
                let (body, boundary) = rebuild_multipart(fields);
                provider::openai::audio_transcription(client, base_url, api_key, body, &boundary)
                    .await
            }
            RemoteProvider::Anthropic { .. } => {
                Err(provider::anthropic::not_implemented("audio_transcription"))
            }
            RemoteProvider::Google { .. } => {
                Err(provider::google::not_implemented("audio_transcription"))
            }
            RemoteProvider::Bedrock { .. } => {
                Err(provider::bedrock::not_implemented("audio_transcription"))
            }
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                let (body, boundary) = rebuild_multipart(fields);
                provider::azure::audio_transcription(
                    client,
                    base_url,
                    api_key,
                    api_version,
                    model,
                    body,
                    &boundary,
                )
                .await
            }
        }
    }

    fn is_openai_compat(&self) -> bool {
        matches!(
            self,
            RemoteProvider::Openai { .. } | RemoteProvider::Azure { .. }
        )
    }

    async fn chat_completion_raw(&self, model: &str, raw_body: Bytes) -> Result<Bytes, Error> {
        match self {
            RemoteProvider::Openai {
                client,
                base_url,
                api_key,
            } => provider::openai::chat_completion_raw(client, base_url, api_key, raw_body).await,
            RemoteProvider::Azure {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                provider::azure::chat_completion_raw(
                    client,
                    base_url,
                    api_key,
                    api_version,
                    model,
                    raw_body,
                )
                .await
            }
            _ => {
                let request: ChatCompletionRequest = serde_json::from_slice(&raw_body)?;
                let resp = self.chat_completion(&request).await?;
                Ok(Bytes::from(serde_json::to_vec(&resp)?))
            }
        }
    }

    fn is_anthropic_compat(&self) -> bool {
        matches!(self, RemoteProvider::Anthropic { .. })
    }

    async fn anthropic_messages_raw(&self, raw_body: Bytes) -> Result<Bytes, Error> {
        match self {
            RemoteProvider::Anthropic { client, api_key } => {
                provider::anthropic::anthropic_messages_raw(client, api_key, raw_body).await
            }
            _ => Err(Error::not_implemented("anthropic_messages_raw")),
        }
    }
}

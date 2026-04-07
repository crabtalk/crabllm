use bytes::Bytes;
use crabllm_core::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, ImageRequest,
    MultipartField, Provider, ProviderConfig, ProviderKind,
};
use futures::stream::StreamExt;
pub use registry::{Deployment, ProviderRegistry};

mod provider;
mod registry;

/// A configured remote-API provider, ready to dispatch requests.
///
/// Each variant carries a `reqwest::Client` so the provider trait
/// implementation needs no shared client passed by the caller. Cloning a
/// `RemoteProvider` is cheap — `reqwest::Client` is internally `Arc`-shared.
#[derive(Debug, Clone)]
pub enum RemoteProvider {
    /// OpenAI-compatible providers (OpenAI, Ollama, vLLM, Groq, etc.).
    /// Request body is forwarded as-is with URL + auth rewrite.
    Openai {
        client: reqwest::Client,
        base_url: String,
        api_key: String,
    },
    /// Anthropic Messages API. Requires request/response translation.
    Anthropic {
        client: reqwest::Client,
        api_key: String,
    },
    /// Google Gemini API. Requires request/response translation.
    Google {
        client: reqwest::Client,
        api_key: String,
    },
    /// AWS Bedrock. Requires SigV4 signing + translation.
    Bedrock {
        client: reqwest::Client,
        region: String,
        access_key: String,
        secret_key: String,
    },
    /// Azure OpenAI. Uses deployment-based URL and api-key header.
    Azure {
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        api_version: String,
    },
}

/// Build the shared `reqwest::Client` used by every `RemoteProvider`
/// instance. `tcp_nodelay(true)` matches the gateway's inbound listener
/// and avoids Nagle-buffering small SSE writes on the proxy → upstream
/// hop. Called once at registry construction and the result is cloned
/// into every `RemoteProvider`, so all providers share a single
/// connection pool, DNS resolver, and TLS state.
pub(crate) fn make_client() -> reqwest::Client {
    reqwest::Client::builder()
        .tcp_nodelay(true)
        .build()
        .expect("crabllm: failed to build reqwest client")
}

impl RemoteProvider {
    /// Build a `RemoteProvider` from a `ProviderConfig`, reusing a shared
    /// `reqwest::Client`. Cloning the client is cheap — internally it's
    /// `Arc<ClientRef>` — so every provider returned by this constructor
    /// dispatches through the same connection pool.
    ///
    /// Routes via [`ProviderConfig::effective_kind`] so a config with
    /// `kind = "openai"` and a `base_url` containing "anthropic" auto-upgrades
    /// to the Anthropic dispatch path.
    pub fn new(config: &ProviderConfig, client: reqwest::Client) -> Self {
        match config.effective_kind() {
            ProviderKind::Openai => RemoteProvider::Openai {
                client,
                base_url: config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
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
                base_url: config
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434/v1".to_string()),
                api_key: config.api_key.clone().unwrap_or_default(),
            },
            ProviderKind::Azure => RemoteProvider::Azure {
                client,
                base_url: config.base_url.clone().unwrap_or_default(),
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
            ProviderKind::LlamaCpp => {
                // Unreachable at runtime: when the `llamacpp` feature is on,
                // `spawn_llamacpp_servers` rewrites the config kind to `Openai`
                // before `from_config` runs. When the feature is off,
                // `validate_provider` rejects this kind before construction.
                // This arm exists only so the match stays exhaustive over
                // `ProviderKind`.
                RemoteProvider::Openai {
                    client,
                    base_url: config.base_url.clone().unwrap_or_default(),
                    api_key: String::new(),
                }
            }
        }
    }
}

/// Build a `reqwest::multipart::Form` from the provider-trait-friendly
/// `MultipartField` representation. Used by the audio-transcription trait
/// impls so each call rebuilds a fresh form (multipart parts are not
/// re-usable across attempts).
fn rebuild_form(fields: &[MultipartField]) -> reqwest::multipart::Form {
    let mut form = reqwest::multipart::Form::new();
    for field in fields {
        let mut part = reqwest::multipart::Part::stream(field.bytes.clone());
        if let Some(ref filename) = field.filename {
            part = part.file_name(filename.clone());
        }
        if let Some(ref content_type) = field.content_type {
            part = part
                .mime_str(content_type)
                .unwrap_or_else(|_| reqwest::multipart::Part::stream(field.bytes.clone()));
        }
        form = form.part(field.name.clone(), part);
    }
    form
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
                let form = rebuild_form(fields);
                provider::openai::audio_transcription(client, base_url, api_key, form).await
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
                let form = rebuild_form(fields);
                provider::azure::audio_transcription(
                    client,
                    base_url,
                    api_key,
                    api_version,
                    model,
                    form,
                )
                .await
            }
        }
    }
}

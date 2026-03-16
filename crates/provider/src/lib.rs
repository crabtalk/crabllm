use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, Error,
};
use futures::stream::{BoxStream, StreamExt};

pub use registry::{Deployment, ProviderRegistry};

mod provider;
mod registry;

/// A configured provider instance, ready to dispatch requests.
#[derive(Debug, Clone)]
pub enum Provider {
    /// OpenAI-compatible providers (OpenAI, Ollama, vLLM, Groq, etc.).
    /// Request body is forwarded as-is with URL + auth rewrite.
    OpenAiCompat { base_url: String, api_key: String },
    /// Anthropic Messages API. Requires request/response translation.
    Anthropic { api_key: String },
    /// Google Gemini API. Requires request/response translation.
    Google { api_key: String },
    /// AWS Bedrock. Requires SigV4 signing + translation.
    Bedrock {
        region: String,
        access_key: String,
        secret_key: String,
    },
    /// Azure OpenAI. Uses deployment-based URL and api-key header.
    Azure {
        base_url: String,
        api_key: String,
        api_version: String,
    },
}

impl Provider {
    /// Send a non-streaming chat completion request.
    pub async fn chat_completion(
        &self,
        client: &reqwest::Client,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        match self {
            Provider::OpenAiCompat { base_url, api_key } => {
                provider::openai::chat_completion(client, base_url, api_key, request).await
            }
            Provider::Anthropic { api_key } => {
                provider::anthropic::chat_completion(client, api_key, request).await
            }
            Provider::Google { api_key } => {
                provider::google::chat_completion(client, api_key, request).await
            }
            #[cfg(feature = "provider-bedrock")]
            Provider::Bedrock {
                region,
                access_key,
                secret_key,
            } => {
                provider::bedrock::chat_completion(client, region, access_key, secret_key, request)
                    .await
            }
            #[cfg(not(feature = "provider-bedrock"))]
            Provider::Bedrock { .. } => Err(provider::bedrock::not_implemented("chat")),
            Provider::Azure {
                base_url,
                api_key,
                api_version,
            } => {
                provider::azure::chat_completion(client, base_url, api_key, api_version, request)
                    .await
            }
        }
    }

    /// Send an embedding request.
    pub async fn embedding(
        &self,
        client: &reqwest::Client,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, Error> {
        match self {
            Provider::OpenAiCompat { base_url, api_key } => {
                provider::openai::embedding(client, base_url, api_key, request).await
            }
            Provider::Anthropic { .. } => Err(provider::anthropic::not_implemented("embedding")),
            Provider::Google { .. } => Err(provider::google::not_implemented("embedding")),
            Provider::Bedrock { .. } => Err(provider::bedrock::not_implemented("embedding")),
            Provider::Azure {
                base_url,
                api_key,
                api_version,
            } => provider::azure::embedding(client, base_url, api_key, api_version, request).await,
        }
    }

    /// Send a streaming chat completion request.
    /// Returns a boxed async stream of parsed SSE chunks.
    pub async fn chat_completion_stream(
        &self,
        client: &reqwest::Client,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        match self {
            Provider::OpenAiCompat { base_url, api_key } => {
                let s =
                    provider::openai::chat_completion_stream(client, base_url, api_key, request)
                        .await?;
                Ok(s.boxed())
            }
            Provider::Anthropic { api_key } => {
                let s = provider::anthropic::chat_completion_stream(
                    client,
                    api_key,
                    request,
                    &request.model,
                )
                .await?;
                Ok(s.boxed())
            }
            Provider::Google { api_key } => {
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
            Provider::Bedrock {
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
            Provider::Bedrock { .. } => Err(provider::bedrock::not_implemented("streaming")),
            Provider::Azure {
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
}

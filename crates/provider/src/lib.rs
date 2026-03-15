use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, Error,
};
use futures::stream::Stream;

pub use registry::ProviderRegistry;

mod openai_compat;
mod registry;

/// A configured provider instance, ready to dispatch requests.
#[derive(Debug, Clone)]
pub enum Provider {
    /// OpenAI-compatible providers (OpenAI, Azure, Ollama, vLLM, Groq, etc.).
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
                openai_compat::chat_completion(client, base_url, api_key, request).await
            }
            Provider::Anthropic { .. } => Err(Error::Internal(
                "anthropic translation not yet implemented".into(),
            )),
            Provider::Google { .. } => Err(Error::Internal(
                "google translation not yet implemented".into(),
            )),
            Provider::Bedrock { .. } => Err(Error::Internal(
                "bedrock translation not yet implemented".into(),
            )),
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
                openai_compat::embedding(client, base_url, api_key, request).await
            }
            Provider::Anthropic { .. } => Err(Error::Internal(
                "anthropic does not support embeddings".into(),
            )),
            Provider::Google { .. } => Err(Error::Internal(
                "google embedding not yet implemented".into(),
            )),
            Provider::Bedrock { .. } => Err(Error::Internal(
                "bedrock embedding not yet implemented".into(),
            )),
        }
    }

    /// Send a streaming chat completion request.
    /// Returns an async stream of parsed SSE chunks.
    pub async fn chat_completion_stream(
        &self,
        client: &reqwest::Client,
        request: &ChatCompletionRequest,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>>, Error> {
        match self {
            Provider::OpenAiCompat { base_url, api_key } => {
                openai_compat::chat_completion_stream(client, base_url, api_key, request).await
            }
            Provider::Anthropic { .. } => Err(Error::Internal(
                "anthropic streaming not yet implemented".into(),
            )),
            Provider::Google { .. } => Err(Error::Internal(
                "google streaming not yet implemented".into(),
            )),
            Provider::Bedrock { .. } => Err(Error::Internal(
                "bedrock streaming not yet implemented".into(),
            )),
        }
    }
}

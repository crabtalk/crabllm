use crate::provider::openai;
use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, Error,
};
use futures::stream::Stream;

/// Build an Azure OpenAI deployment URL.
/// Format: {base_url}/openai/deployments/{model}/{path}?api-version={api_version}
fn azure_url(base_url: &str, model: &str, path: &str, api_version: &str) -> String {
    format!(
        "{}/openai/deployments/{}/{}?api-version={}",
        base_url.trim_end_matches('/'),
        model,
        path,
        api_version,
    )
}

/// Send a non-streaming chat completion to an Azure OpenAI deployment.
pub async fn chat_completion(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let url = azure_url(base_url, &request.model, "chat/completions", api_version);
    let resp = client
        .post(&url)
        .header("api-key", api_key)
        .json(request)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    resp.json::<ChatCompletionResponse>()
        .await
        .map_err(|e| Error::Internal(e.to_string()))
}

/// Send an embedding request to an Azure OpenAI deployment.
pub async fn embedding(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, Error> {
    let url = azure_url(base_url, &request.model, "embeddings", api_version);
    let resp = client
        .post(&url)
        .header("api-key", api_key)
        .json(request)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    resp.json::<EmbeddingResponse>()
        .await
        .map_err(|e| Error::Internal(e.to_string()))
}

/// Send a streaming chat completion to an Azure OpenAI deployment.
/// Reuses the OpenAI SSE parser since Azure streams in the same format.
pub async fn chat_completion_stream(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let url = azure_url(base_url, &request.model, "chat/completions", api_version);
    let resp = client
        .post(&url)
        .header("api-key", api_key)
        .json(request)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    Ok(openai::sse_stream(resp))
}

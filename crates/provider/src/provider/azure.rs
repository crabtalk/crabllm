use crate::HttpClient;
use crate::provider::openai;
use bytes::Bytes;
use crabllm_core::{
    AudioSpeechRequest, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    EmbeddingRequest, EmbeddingResponse, Error, ImageRequest,
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
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let url = azure_url(base_url, &request.model, "chat/completions", api_version);
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [("content-type", "application/json"), ("api-key", api_key)];
    let resp = client
        .post(&url, &headers, body.into())
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
        });
    }

    sonic_rs::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))
}

/// Forward raw JSON bytes to an Azure OpenAI chat completions deployment,
/// returning the response bytes without deserialization.
pub async fn chat_completion_raw(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    model: &str,
    raw_body: Bytes,
) -> Result<Bytes, Error> {
    let url = azure_url(base_url, model, "chat/completions", api_version);
    let headers = [("content-type", "application/json"), ("api-key", api_key)];
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

/// Send an embedding request to an Azure OpenAI deployment.
pub async fn embedding(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, Error> {
    let url = azure_url(base_url, &request.model, "embeddings", api_version);
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [("content-type", "application/json"), ("api-key", api_key)];
    let resp = client
        .post(&url, &headers, body.into())
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
        });
    }

    sonic_rs::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))
}

/// Send an image generation request to an Azure OpenAI deployment.
pub async fn image_generation(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &ImageRequest,
) -> Result<(Bytes, String), Error> {
    let url = azure_url(base_url, &request.model, "images/generations", api_version);
    raw_pass_through(client, &url, api_key, request).await
}

/// Send a text-to-speech request to an Azure OpenAI deployment.
pub async fn audio_speech(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &AudioSpeechRequest,
) -> Result<(Bytes, String), Error> {
    let url = azure_url(base_url, &request.model, "audio/speech", api_version);
    let (bytes, content_type) = raw_pass_through(client, &url, api_key, request).await?;
    let content_type = if content_type == "application/json" {
        "audio/mpeg".to_string()
    } else {
        content_type
    };
    Ok((bytes, content_type))
}

/// Forward a JSON request to Azure and return raw response bytes + content-type.
pub(crate) async fn raw_pass_through<T: serde::Serialize>(
    client: &HttpClient,
    url: &str,
    api_key: &str,
    request: &T,
) -> Result<(Bytes, String), Error> {
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [("content-type", "application/json"), ("api-key", api_key)];
    let resp = client
        .post(url, &headers, body.into())
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
        });
    }

    let content_type = resp
        .content_type
        .unwrap_or_else(|| "application/json".to_string());
    Ok((resp.body, content_type))
}

/// Send an audio transcription request to an Azure OpenAI deployment.
/// Takes model separately since the multipart form is opaque.
pub async fn audio_transcription(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    model: &str,
    body: Bytes,
    boundary: &str,
) -> Result<(Bytes, String), Error> {
    let url = azure_url(base_url, model, "audio/transcriptions", api_version);
    let content_type_header = format!("multipart/form-data; boundary={boundary}");
    let headers = [
        ("content-type", content_type_header.as_str()),
        ("api-key", api_key),
    ];
    let resp = client
        .post(&url, &headers, body)
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
        });
    }

    let content_type = resp
        .content_type
        .unwrap_or_else(|| "application/json".to_string());
    Ok((resp.body, content_type))
}

/// Send a streaming chat completion to an Azure OpenAI deployment.
/// Reuses the OpenAI SSE parser since Azure streams in the same format.
pub async fn chat_completion_stream(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    api_version: &str,
    request: &ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let url = azure_url(base_url, &request.model, "chat/completions", api_version);
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [("content-type", "application/json"), ("api-key", api_key)];
    let byte_stream = client.post_stream(&url, &headers, body.into()).await?;

    Ok(openai::sse_stream(byte_stream))
}

use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, Error,
};
use futures::stream::{self, Stream};
use reqwest::Response;

/// Send a non-streaming chat completion to an OpenAI-compatible endpoint.
pub async fn chat_completion(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
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

/// Send an embedding request to an OpenAI-compatible endpoint.
pub async fn embedding(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, Error> {
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
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

/// Send a streaming chat completion to an OpenAI-compatible endpoint.
/// Returns an async stream of parsed SSE chunks.
pub async fn chat_completion_stream(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .json(request)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    Ok(sse_stream(resp))
}

/// Parse an SSE byte stream into `ChatCompletionChunk` items.
pub(crate) fn sse_stream(resp: Response) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    let byte_stream = resp.bytes_stream();

    stream::unfold(
        (byte_stream, String::new()),
        |(mut byte_stream, mut buffer)| async move {
            use futures::TryStreamExt;

            loop {
                // Try to extract a complete line from the buffer.
                if let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        let data = data.trim();
                        if data == "[DONE]" {
                            return None;
                        }
                        match serde_json::from_str::<ChatCompletionChunk>(data) {
                            Ok(chunk) => return Some((Ok(chunk), (byte_stream, buffer))),
                            Err(e) => {
                                return Some((
                                    Err(Error::Internal(format!("SSE parse error: {e}"))),
                                    (byte_stream, buffer),
                                ));
                            }
                        }
                    }
                    // Skip non-data lines (comments, event:, etc.)
                    continue;
                }

                // Need more data from the stream.
                match byte_stream.try_next().await {
                    Ok(Some(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Ok(None) => return None,
                    Err(e) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buffer),
                        ));
                    }
                }
            }
        },
    )
}

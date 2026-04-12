use crate::{ByteStream, HttpClient};
use bytes::{Buf, Bytes, BytesMut};
use crabllm_core::{
    AudioSpeechRequest, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    EmbeddingRequest, EmbeddingResponse, Error, ImageRequest,
};
use futures::stream::{self, Stream};

/// Send a non-streaming chat completion to an OpenAI-compatible endpoint.
pub async fn chat_completion(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
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

/// Send an embedding request to an OpenAI-compatible endpoint.
pub async fn embedding(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, Error> {
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
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

/// Forward raw JSON bytes to an OpenAI-compatible chat completions
/// endpoint, returning the response bytes without deserialization.
pub async fn chat_completion_raw(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<Bytes, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
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

/// Send a streaming chat completion to an OpenAI-compatible endpoint.
/// Returns an async stream of parsed SSE chunks.
pub async fn chat_completion_stream(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    let byte_stream = client.post_stream(&url, &headers, body.into()).await?;

    Ok(sse_stream(byte_stream))
}

/// Send an image generation request to an OpenAI-compatible endpoint.
/// Returns raw response bytes and content-type header.
pub async fn image_generation(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &ImageRequest,
) -> Result<(Bytes, String), Error> {
    let url = format!("{}/images/generations", base_url.trim_end_matches('/'));
    raw_pass_through(client, &url, api_key, request).await
}

/// Send a text-to-speech request to an OpenAI-compatible endpoint.
/// Returns raw audio bytes and content-type header.
pub async fn audio_speech(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &AudioSpeechRequest,
) -> Result<(Bytes, String), Error> {
    let url = format!("{}/audio/speech", base_url.trim_end_matches('/'));
    let (bytes, content_type) = raw_pass_through(client, &url, api_key, request).await?;
    // Default to audio/mpeg if upstream omits Content-Type.
    let content_type = if content_type == "application/json" {
        "audio/mpeg".to_string()
    } else {
        content_type
    };
    Ok((bytes, content_type))
}

/// Forward a JSON request and return raw response bytes + content-type.
pub(crate) async fn raw_pass_through<T: serde::Serialize>(
    client: &HttpClient,
    url: &str,
    api_key: &str,
    request: &T,
) -> Result<(Bytes, String), Error> {
    let body = sonic_rs::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
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

/// Send an audio transcription request to an OpenAI-compatible endpoint.
/// Takes pre-built multipart body bytes and boundary. Returns raw response bytes + content-type.
pub async fn audio_transcription(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    body: Bytes,
    boundary: &str,
) -> Result<(Bytes, String), Error> {
    let url = format!("{}/audio/transcriptions", base_url.trim_end_matches('/'));
    let content_type_header = format!("multipart/form-data; boundary={boundary}");
    let headers = [
        ("content-type", content_type_header.as_str()),
        ("authorization", &format!("Bearer {api_key}")),
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

/// Parse an SSE byte stream into `ChatCompletionChunk` items.
pub(crate) fn sse_stream(
    byte_stream: ByteStream,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    stream::unfold(
        (byte_stream, BytesMut::new()),
        |(mut byte_stream, mut buffer)| async move {
            use futures::StreamExt;

            loop {
                if let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                    let mut line_end = newline_pos;
                    if line_end > 0 && buffer[line_end - 1] == b'\r' {
                        line_end -= 1;
                    }
                    let line = &buffer[..line_end];

                    if line.is_empty() {
                        buffer.advance(newline_pos + 1);
                        continue;
                    }

                    if let Some(data) = line.strip_prefix(b"data: ") {
                        let data = match std::str::from_utf8(data) {
                            Ok(s) => s.trim(),
                            Err(_) => {
                                buffer.advance(newline_pos + 1);
                                continue;
                            }
                        };
                        if data == "[DONE]" {
                            return None;
                        }
                        let result = match sonic_rs::from_str::<ChatCompletionChunk>(data) {
                            Ok(chunk) => Ok(chunk),
                            Err(e) => Err(Error::Internal(format!("SSE parse error: {e}"))),
                        };
                        buffer.advance(newline_pos + 1);
                        return Some((result, (byte_stream, buffer)));
                    }
                    // Skip non-data lines (comments, event:, etc.)
                    buffer.advance(newline_pos + 1);
                    continue;
                }

                // Need more data from the stream.
                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.extend_from_slice(&bytes);
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buffer),
                        ));
                    }
                    None => return None,
                }
            }
        },
    )
}

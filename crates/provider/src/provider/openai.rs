use crate::{ByteStream, HttpClient};
use bytes::Bytes;
use crabllm_core::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, ImageRequest, ModelList,
    MultipartField, Provider, anthropic,
};
use futures::stream::{Stream, StreamExt};

#[derive(Debug, Clone)]
pub struct OpenaiProvider {
    pub(crate) client: HttpClient,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
}

impl Provider for OpenaiProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        chat_completion(&self.client, &self.base_url, &self.api_key, request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let s =
            chat_completion_stream(&self.client, &self.base_url, &self.api_key, request).await?;
        Ok(s.boxed())
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        embedding(&self.client, &self.base_url, &self.api_key, request).await
    }

    async fn models(&self) -> Result<ModelList, Error> {
        models(&self.client, &self.base_url, &self.api_key).await
    }

    async fn image_generation(&self, request: &ImageRequest) -> Result<(Bytes, String), Error> {
        image_generation(&self.client, &self.base_url, &self.api_key, request).await
    }

    async fn audio_speech(&self, request: &AudioSpeechRequest) -> Result<(Bytes, String), Error> {
        audio_speech(&self.client, &self.base_url, &self.api_key, request).await
    }

    async fn audio_transcription(
        &self,
        _model: &str,
        fields: &[MultipartField],
    ) -> Result<(Bytes, String), Error> {
        let (body, boundary) = crate::rebuild_multipart(fields);
        audio_transcription(&self.client, &self.base_url, &self.api_key, body, &boundary).await
    }

    async fn anthropic_messages(
        &self,
        request: &anthropic::Request,
    ) -> Result<anthropic::Response, Error> {
        let ir_resp = self
            .complete(&crabllm_core::ir::Request::from(request.clone()))
            .await?;
        Ok(anthropic::Response::from(&ir_resp))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &anthropic::Request,
    ) -> Result<BoxStream<'static, Result<anthropic::StreamEvent, Error>>, Error> {
        crate::anthropic_stream_via_chat(self, request).await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &crabllm_core::gemini::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::gemini::Response, Error>>, Error> {
        crate::gemini_stream_via_chat(self, model, request).await
    }

    async fn complete(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<crabllm_core::ir::Response, Error> {
        crate::complete_via_chat(self, request).await
    }

    async fn complete_stream(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ir::StreamEvent, Error>>, Error> {
        crate::complete_stream_via_chat(self, request).await
    }

    fn is_openai_compat(&self) -> bool {
        true
    }

    async fn chat_completion_stream_passthrough(
        &self,
        _model: &str,
        body_stream: crabllm_core::ByteStream,
    ) -> Result<crabllm_core::ByteStream, Error> {
        chat_completion_stream_passthrough(&self.client, &self.base_url, &self.api_key, body_stream)
            .await
    }

    async fn chat_completion_stream_raw(
        &self,
        _model: &str,
        raw_body: Bytes,
    ) -> Result<crabllm_core::ByteStream, Error> {
        chat_completion_stream_raw(&self.client, &self.base_url, &self.api_key, raw_body).await
    }

    async fn chat_completion_raw(&self, _model: &str, raw_body: Bytes) -> Result<Bytes, Error> {
        chat_completion_raw(&self.client, &self.base_url, &self.api_key, raw_body).await
    }
}

/// Send a non-streaming chat completion to an OpenAI-compatible endpoint.
pub async fn chat_completion(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    let resp = client
        .post(&url, &headers, body.into())
        .await?
        .error_for_status()?;

    crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))
}

/// List the models an OpenAI-compatible endpoint exposes. Third parties send
/// leaner rows than the gateway does — `id` is the only field guaranteed.
pub async fn models(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
) -> Result<ModelList, Error> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    let resp = client.get(&url, &headers).await?.error_for_status()?;

    crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))
}

/// Send an embedding request to an OpenAI-compatible endpoint.
pub async fn embedding(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, Error> {
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    let resp = client
        .post(&url, &headers, body.into())
        .await?
        .error_for_status()?;

    crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))
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
        .await?
        .error_for_status()?;

    Ok(resp.body)
}

/// Stream a client body straight through to an OpenAI-compatible chat
/// completions endpoint. The body is consumed as it arrives — no buffering,
/// so no retries.
pub async fn chat_completion_stream_passthrough(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    body_stream: ByteStream,
) -> Result<ByteStream, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    client.post_stream_body(&url, &headers, body_stream).await
}

/// Stream raw SSE bytes from an OpenAI-compatible chat completions endpoint.
pub async fn chat_completion_stream_raw(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<ByteStream, Error> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    client.post_stream(&url, &headers, raw_body).await
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
    let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    let byte_stream = client.post_stream(&url, &headers, body.into()).await?;

    Ok(crabllm_core::codec::openai::sse_stream(byte_stream))
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
    let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
    let headers = [
        ("content-type", "application/json"),
        ("authorization", &format!("Bearer {api_key}")),
    ];
    let resp = client
        .post(url, &headers, body.into())
        .await?
        .error_for_status()?;

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
        .await?
        .error_for_status()?;

    let content_type = resp
        .content_type
        .unwrap_or_else(|| "application/json".to_string());
    Ok((resp.body, content_type))
}

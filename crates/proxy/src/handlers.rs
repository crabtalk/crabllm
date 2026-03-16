use crate::{AppState, auth::KeyName};
use axum::{
    Extension, Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use crabtalk_core::{
    ApiError, AudioSpeechRequest, ChatCompletionRequest, EmbeddingRequest, ImageRequest, Model,
    ModelList, RequestContext, Storage,
};
use crabtalk_provider::Deployment;
use futures::StreamExt;
use rand::Rng;
use std::time::{Duration, Instant};

/// POST /v1/chat/completions
pub async fn chat_completions<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let model = state.registry.resolve(&request.model).to_string();
    let deployments = match state.registry.dispatch_list(&model) {
        Some(list) => list,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{model}' not found"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        key_name: key_name.0,
        is_stream: request.stream == Some(true),
        started_at: Instant::now(),
    };

    // Run on_request hooks — short-circuit on first error.
    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return (
                StatusCode::from_u16(ext_err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(ext_err.body),
            )
                .into_response();
        }
    }

    if ctx.is_stream {
        // Streaming: retry + fallback on connection errors only (pre-stream).
        let mut last_err = None;
        for deployment in &deployments {
            match try_stream_with_retries(deployment, &state.client, &request).await {
                Ok(stream) => {
                    let extensions = state.extensions.clone();
                    let ctx_clone = ctx.clone();

                    let observed = stream.then(move |result| {
                        let extensions = extensions.clone();
                        let ctx = ctx_clone.clone();
                        async move {
                            match &result {
                                Ok(chunk) => {
                                    for ext in extensions.iter() {
                                        ext.on_chunk(&ctx, chunk).await;
                                    }
                                }
                                Err(error) => {
                                    for ext in extensions.iter() {
                                        ext.on_error(&ctx, error).await;
                                    }
                                }
                            }
                            result
                        }
                    });

                    let sse_stream = observed.map(|result| match result {
                        Ok(chunk) => {
                            let json = serde_json::to_string(&chunk).unwrap_or_default();
                            Ok(Event::default().data(json))
                        }
                        Err(e) => {
                            let json = serde_json::to_string(&ApiError::new(
                                e.to_string(),
                                "server_error",
                            ))
                            .unwrap_or_default();
                            Ok(Event::default().data(json))
                        }
                    });

                    let done = futures::stream::once(async {
                        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                    });
                    let full_stream = sse_stream.chain(done);

                    return Sse::new(full_stream)
                        .keep_alive(axum::response::sse::KeepAlive::new())
                        .into_response();
                }
                Err(e) => last_err = Some(e),
            }
        }

        let e = last_err.unwrap_or_else(|| {
            crabtalk_core::Error::Internal("no providers available".to_string())
        });
        for ext in state.extensions.iter() {
            ext.on_error(&ctx, &e).await;
        }
        error_response(e)
    } else {
        // Non-streaming: check cache first.
        for ext in state.extensions.iter() {
            if let Some(cached) = ext.on_cache_lookup(&request).await {
                return Json(cached).into_response();
            }
        }

        let mut last_err = None;
        for deployment in &deployments {
            match try_chat_with_retries(deployment, &state.client, &request).await {
                Ok(resp) => {
                    for ext in state.extensions.iter() {
                        ext.on_response(&ctx, &request, &resp).await;
                    }
                    return Json(resp).into_response();
                }
                Err(e) => last_err = Some(e),
            }
        }

        let e = last_err.unwrap_or_else(|| {
            crabtalk_core::Error::Internal("no providers available".to_string())
        });
        for ext in state.extensions.iter() {
            ext.on_error(&ctx, &e).await;
        }
        error_response(e)
    }
}

/// POST /v1/embeddings
pub async fn embeddings<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<EmbeddingRequest>,
) -> Response {
    let model = state.registry.resolve(&request.model).to_string();
    let deployments = match state.registry.dispatch_list(&model) {
        Some(list) => list,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{model}' not found"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        key_name: key_name.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    // Run on_request hooks — short-circuit on first error.
    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return (
                StatusCode::from_u16(ext_err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(ext_err.body),
            )
                .into_response();
        }
    }

    let mut last_err = None;
    for deployment in &deployments {
        match try_embedding_with_retries(deployment, &state.client, &request).await {
            Ok(resp) => return Json(resp).into_response(),
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabtalk_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    error_response(e)
}

/// GET /v1/models
pub async fn models<S: Storage + 'static>(State(state): State<AppState<S>>) -> Json<ModelList> {
    let data: Vec<Model> = state
        .config
        .models()
        .into_keys()
        .map(|name| Model {
            id: name,
            object: "model".to_string(),
            created: 0,
            owned_by: "crabtalk".to_string(),
        })
        .collect();

    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}

/// POST /v1/images/generations
pub async fn image_generations<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<ImageRequest>,
) -> Response {
    let model = state.registry.resolve(&request.model).to_string();
    let deployments = match state.registry.dispatch_list(&model) {
        Some(list) => list,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{model}' not found"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        key_name: key_name.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return (
                StatusCode::from_u16(ext_err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(ext_err.body),
            )
                .into_response();
        }
    }

    // Fallback only — no retry (image generation is non-idempotent + billed).
    let mut last_err = None;
    for deployment in &deployments {
        match with_timeout(
            deployment.timeout,
            deployment
                .provider
                .image_generation(&state.client, &request),
        )
        .await
        {
            Ok((bytes, content_type)) => {
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabtalk_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    error_response(e)
}

/// POST /v1/audio/speech
pub async fn audio_speech<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<AudioSpeechRequest>,
) -> Response {
    let model = state.registry.resolve(&request.model).to_string();
    let deployments = match state.registry.dispatch_list(&model) {
        Some(list) => list,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{model}' not found"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        key_name: key_name.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return (
                StatusCode::from_u16(ext_err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(ext_err.body),
            )
                .into_response();
        }
    }

    // Fallback only — no retry (TTS is non-idempotent).
    let mut last_err = None;
    for deployment in &deployments {
        match with_timeout(
            deployment.timeout,
            deployment.provider.audio_speech(&state.client, &request),
        )
        .await
        {
            Ok((bytes, content_type)) => {
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabtalk_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    error_response(e)
}

/// A buffered multipart field for reconstructing forms across fallback attempts.
struct BufferedField {
    name: String,
    filename: Option<String>,
    content_type: Option<String>,
    bytes: bytes::Bytes,
}

/// Rebuild a `reqwest::multipart::Form` from buffered fields.
fn rebuild_form(fields: &[BufferedField]) -> reqwest::multipart::Form {
    let mut form = reqwest::multipart::Form::new();
    for field in fields {
        let mut part = reqwest::multipart::Part::bytes(field.bytes.to_vec());
        if let Some(ref filename) = field.filename {
            part = part.file_name(filename.clone());
        }
        if let Some(ref content_type) = field.content_type {
            part = part
                .mime_str(content_type)
                .unwrap_or_else(|_| reqwest::multipart::Part::bytes(field.bytes.to_vec()));
        }
        form = form.part(field.name.clone(), part);
    }
    form
}

/// POST /v1/audio/transcriptions
pub async fn audio_transcriptions<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    mut multipart: Multipart,
) -> Response {
    // Buffer all multipart fields and extract the model name.
    let mut fields = Vec::new();
    let mut model_value = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = match field.name() {
            Some(n) => n.to_string(),
            None => continue,
        };
        let filename = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(
                        format!("failed to read multipart field: {e}"),
                        "invalid_request_error",
                    )),
                )
                    .into_response();
            }
        };

        if name == "model" {
            model_value = Some(String::from_utf8_lossy(&bytes).to_string());
        }
        fields.push(BufferedField {
            name,
            filename,
            content_type,
            bytes,
        });
    }

    let model = match model_value {
        Some(m) => state.registry.resolve(&m).to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    "missing 'model' field in multipart form",
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let deployments = match state.registry.dispatch_list(&model) {
        Some(list) => list,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{model}' not found"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        key_name: key_name.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return (
                StatusCode::from_u16(ext_err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(ext_err.body),
            )
                .into_response();
        }
    }

    // Fallback only — rebuild form for each attempt.
    let mut last_err = None;
    for deployment in &deployments {
        let form = rebuild_form(&fields);
        match with_timeout(
            deployment.timeout,
            deployment
                .provider
                .audio_transcription(&state.client, &model, form),
        )
        .await
        {
            Ok((bytes, content_type)) => {
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabtalk_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    error_response(e)
}

/// Call a provider with a timeout, converting elapsed to Error::Timeout.
/// A zero duration disables the timeout.
async fn with_timeout<F, T>(timeout: Duration, fut: F) -> Result<T, crabtalk_core::Error>
where
    F: std::future::Future<Output = Result<T, crabtalk_core::Error>>,
{
    if timeout.is_zero() {
        return fut.await;
    }
    match tokio::time::timeout(timeout, fut).await {
        Ok(result) => result,
        Err(_elapsed) => Err(crabtalk_core::Error::Timeout),
    }
}

/// Apply full jitter: random duration in [backoff/2, backoff].
fn jittered(backoff: Duration) -> Duration {
    let lo = backoff / 2;
    rand::rng().random_range(lo..=backoff)
}

/// Retry a non-streaming chat completion on a single deployment.
async fn try_chat_with_retries(
    deployment: &Deployment,
    client: &reqwest::Client,
    request: &ChatCompletionRequest,
) -> Result<crabtalk_core::ChatCompletionResponse, crabtalk_core::Error> {
    let mut last_err;
    match with_timeout(
        deployment.timeout,
        deployment.provider.chat_completion(client, request),
    )
    .await
    {
        Ok(resp) => return Ok(resp),
        Err(e) => {
            if !e.is_transient() || deployment.max_retries == 0 {
                return Err(e);
            }
            last_err = e;
        }
    }

    let mut backoff = Duration::from_millis(100);
    for _ in 0..deployment.max_retries {
        tokio::time::sleep(jittered(backoff)).await;
        backoff *= 2;
        match with_timeout(
            deployment.timeout,
            deployment.provider.chat_completion(client, request),
        )
        .await
        {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                if !e.is_transient() {
                    return Err(e);
                }
                last_err = e;
            }
        }
    }

    Err(last_err)
}

/// Retry a streaming chat completion on a single deployment.
async fn try_stream_with_retries(
    deployment: &Deployment,
    client: &reqwest::Client,
    request: &ChatCompletionRequest,
) -> Result<
    futures::stream::BoxStream<
        'static,
        Result<crabtalk_core::ChatCompletionChunk, crabtalk_core::Error>,
    >,
    crabtalk_core::Error,
> {
    let mut last_err;
    match with_timeout(
        deployment.timeout,
        deployment.provider.chat_completion_stream(client, request),
    )
    .await
    {
        Ok(stream) => return Ok(stream),
        Err(e) => {
            if !e.is_transient() || deployment.max_retries == 0 {
                return Err(e);
            }
            last_err = e;
        }
    }

    let mut backoff = Duration::from_millis(100);
    for _ in 0..deployment.max_retries {
        tokio::time::sleep(jittered(backoff)).await;
        backoff *= 2;
        match with_timeout(
            deployment.timeout,
            deployment.provider.chat_completion_stream(client, request),
        )
        .await
        {
            Ok(stream) => return Ok(stream),
            Err(e) => {
                if !e.is_transient() {
                    return Err(e);
                }
                last_err = e;
            }
        }
    }

    Err(last_err)
}

/// Retry an embedding request on a single deployment.
async fn try_embedding_with_retries(
    deployment: &Deployment,
    client: &reqwest::Client,
    request: &EmbeddingRequest,
) -> Result<crabtalk_core::EmbeddingResponse, crabtalk_core::Error> {
    let mut last_err;
    match with_timeout(
        deployment.timeout,
        deployment.provider.embedding(client, request),
    )
    .await
    {
        Ok(resp) => return Ok(resp),
        Err(e) => {
            if !e.is_transient() || deployment.max_retries == 0 {
                return Err(e);
            }
            last_err = e;
        }
    }

    let mut backoff = Duration::from_millis(100);
    for _ in 0..deployment.max_retries {
        tokio::time::sleep(jittered(backoff)).await;
        backoff *= 2;
        match with_timeout(
            deployment.timeout,
            deployment.provider.embedding(client, request),
        )
        .await
        {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                if !e.is_transient() {
                    return Err(e);
                }
                last_err = e;
            }
        }
    }

    Err(last_err)
}

/// Map a provider Error to an HTTP error response.
fn error_response(e: crabtalk_core::Error) -> Response {
    let (status, api_error) = match &e {
        crabtalk_core::Error::Provider { status, body } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            ApiError::new(body.clone(), "upstream_error"),
        ),
        crabtalk_core::Error::Timeout => (
            StatusCode::GATEWAY_TIMEOUT,
            ApiError::new(e.to_string(), "timeout_error"),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::new(e.to_string(), "server_error"),
        ),
    };
    (status, Json(api_error)).into_response()
}

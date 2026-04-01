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
use crabllm_core::{
    ApiError, AudioSpeechRequest, ChatCompletionRequest, EmbeddingRequest, ImageRequest, Model,
    ModelList, RequestContext, Storage,
};
use crabllm_provider::Deployment;
use futures::StreamExt;
use rand::Rng;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

fn record_duration(ctx: &RequestContext, status: &'static str) {
    metrics::histogram!("crabllm_request_duration_seconds",
        "provider" => ctx.provider.clone(),
        "model" => ctx.model.clone(),
        "status" => status,
        "stream" => if ctx.is_stream { "true" } else { "false" },
    )
    .record(ctx.started_at.elapsed().as_secs_f64());
}

fn error_status(e: &crabllm_core::Error) -> &'static str {
    match e {
        crabllm_core::Error::Provider { status, .. } => match status {
            429 => "429",
            400..=499 => "4xx",
            _ => "5xx",
        },
        _ => "5xx",
    }
}

fn record_tokens(ctx: &RequestContext, prompt: u32, completion: u32) {
    if prompt > 0 {
        metrics::counter!("crabllm_tokens_total",
            "provider" => ctx.provider.clone(),
            "model" => ctx.model.clone(),
            "direction" => "prompt",
        )
        .increment(prompt as u64);
    }
    if completion > 0 {
        metrics::counter!("crabllm_tokens_total",
            "provider" => ctx.provider.clone(),
            "model" => ctx.model.clone(),
            "direction" => "completion",
        )
        .increment(completion as u64);
    }
}

/// POST /v1/chat/completions
pub async fn chat_completions<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(mut request): Json<ChatCompletionRequest>,
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
        .registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

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
        // Ensure OpenAI-compatible providers include token usage in the final
        // streaming chunk. Harmlessly ignored by Anthropic/Google/Bedrock which
        // build their own request format and don't read `extra`.
        request
            .extra
            .entry("stream_options".to_string())
            .or_insert(serde_json::json!({ "include_usage": true }));

        // Streaming: retry + fallback on connection errors only (pre-stream).
        let mut last_err = None;
        for deployment in &deployments {
            match try_stream_with_retries(deployment, &state.client, &request).await {
                Ok(stream) => {
                    let extensions = state.extensions.clone();
                    let ctx = Arc::new(ctx);
                    let errored = Arc::new(AtomicBool::new(false));

                    let ctx_done = ctx.clone();
                    let errored_done = errored.clone();

                    let observed = stream.then(move |result| {
                        let extensions = extensions.clone();
                        let ctx = ctx.clone();
                        let errored = errored.clone();
                        async move {
                            match &result {
                                Ok(chunk) => {
                                    // Token usage arrives in the final chunk when
                                    // the provider supports stream_options.include_usage.
                                    if let Some(ref usage) = chunk.usage {
                                        record_tokens(
                                            &ctx,
                                            usage.prompt_tokens,
                                            usage.completion_tokens,
                                        );
                                    }
                                    for ext in extensions.iter() {
                                        ext.on_chunk(&ctx, chunk).await;
                                    }
                                }
                                Err(error) => {
                                    errored.store(true, Ordering::Relaxed);
                                    for ext in extensions.iter() {
                                        ext.on_error(&ctx, error).await;
                                    }
                                }
                            }
                            result
                        }
                    });

                    let sse_stream = observed.map(|result| match result {
                        Ok(mut chunk) => {
                            let json = chunk.raw_json.take().unwrap_or_else(|| {
                                serde_json::to_string(&chunk).unwrap_or_default()
                            });
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

                    // Record duration once when the stream terminates.
                    let done = futures::stream::once(async move {
                        let status = if errored_done.load(Ordering::Relaxed) {
                            "5xx"
                        } else {
                            "2xx"
                        };
                        record_duration(&ctx_done, status);
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

        let e = last_err
            .unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".to_string()));
        for ext in state.extensions.iter() {
            ext.on_error(&ctx, &e).await;
        }
        record_duration(&ctx, "5xx");
        error_response(e)
    } else {
        // Non-streaming: check cache first.
        // Cache hits skip duration recording — sub-millisecond responses
        // would skew the histogram, which should reflect provider latency.
        for ext in state.extensions.iter() {
            if let Some(cached) = ext.on_cache_lookup(&request).await {
                return Json(cached).into_response();
            }
        }

        let mut last_err = None;
        for deployment in &deployments {
            match try_chat_with_retries(deployment, &state.client, &request).await {
                Ok(resp) => {
                    if let Some(ref usage) = resp.usage {
                        record_tokens(&ctx, usage.prompt_tokens, usage.completion_tokens);
                    }
                    record_duration(&ctx, "2xx");
                    for ext in state.extensions.iter() {
                        ext.on_response(&ctx, &request, &resp).await;
                    }
                    return Json(resp).into_response();
                }
                Err(e) => last_err = Some(e),
            }
        }

        let e = last_err
            .unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".to_string()));
        for ext in state.extensions.iter() {
            ext.on_error(&ctx, &e).await;
        }
        record_duration(&ctx, error_status(&e));
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
        .registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

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
            Ok(resp) => {
                record_duration(&ctx, "2xx");
                return Json(resp).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
    error_response(e)
}

/// GET /v1/models
pub async fn models<S: Storage + 'static>(State(state): State<AppState<S>>) -> Json<ModelList> {
    let data: Vec<Model> = state
        .registry
        .model_names()
        .map(|name| Model {
            id: name.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "crabllm".to_string(),
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
        .registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

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
                record_duration(&ctx, "2xx");
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
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
        .registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

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
                record_duration(&ctx, "2xx");
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
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

/// POST /v1/audio/transcriptions
pub async fn audio_transcriptions<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    mut multipart: Multipart,
) -> Response {
    // Buffer all multipart fields and extract the model name.
    let mut fields = Vec::with_capacity(8);
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
            model_value = Some(String::from_utf8_lossy(&bytes).into_owned());
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
        .registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

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
                record_duration(&ctx, "2xx");
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
    error_response(e)
}

/// Call a provider with a timeout, converting elapsed to Error::Timeout.
/// A zero duration disables the timeout.
async fn with_timeout<F, T>(timeout: Duration, fut: F) -> Result<T, crabllm_core::Error>
where
    F: std::future::Future<Output = Result<T, crabllm_core::Error>>,
{
    if timeout.is_zero() {
        return fut.await;
    }
    match tokio::time::timeout(timeout, fut).await {
        Ok(result) => result,
        Err(_elapsed) => Err(crabllm_core::Error::Timeout),
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
) -> Result<crabllm_core::ChatCompletionResponse, crabllm_core::Error> {
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
        Result<crabllm_core::ChatCompletionChunk, crabllm_core::Error>,
    >,
    crabllm_core::Error,
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
) -> Result<crabllm_core::EmbeddingResponse, crabllm_core::Error> {
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
fn error_response(e: crabllm_core::Error) -> Response {
    let (status, api_error) = match &e {
        crabllm_core::Error::Provider { status, body } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            ApiError::new(body.clone(), "upstream_error"),
        ),
        crabllm_core::Error::Timeout => (
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

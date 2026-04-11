use crate::{AppState, auth::KeyName, state::UsageEvent};
use axum::{
    Extension, Json,
    extract::{Multipart, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use crabllm_core::{
    ApiError, AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    EmbeddingRequest, ImageRequest, Model, ModelList, MultipartField, Provider, RequestContext,
    Storage,
};
use crabllm_provider::Deployment;
use futures::StreamExt;
use rand::Rng;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use std::time::{Duration, Instant, SystemTime};

pub(crate) fn record_duration(ctx: &RequestContext, status: &'static str) {
    metrics::histogram!("crabllm_request_duration_seconds",
        "provider" => ctx.provider.clone(),
        "model" => ctx.model.clone(),
        "status" => status,
        "stream" => if ctx.is_stream { "true" } else { "false" },
    )
    .record(ctx.started_at.elapsed().as_secs_f64());
}

pub(crate) fn error_status(e: &crabllm_core::Error) -> &'static str {
    match e {
        crabllm_core::Error::Provider { status, .. } => match status {
            429 => "429",
            400..=499 => "4xx",
            _ => "5xx",
        },
        _ => "5xx",
    }
}

pub(crate) fn record_tokens(ctx: &RequestContext, prompt: u32, completion: u32) {
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

pub(crate) fn emit_usage<S: Storage, P: Provider>(
    state: &AppState<S, P>,
    ctx: &RequestContext,
    endpoint: &'static str,
    tokens_in: u32,
    tokens_out: u32,
    status: u16,
    error: Option<String>,
) {
    let Some(tx) = state.usage_events.as_ref() else {
        return;
    };
    let _ = tx.send(UsageEvent {
        timestamp: SystemTime::now(),
        request_id: ctx.request_id.clone(),
        key_name: ctx.key_name.clone(),
        model: ctx.model.clone(),
        provider: ctx.provider.clone(),
        endpoint,
        tokens_in,
        tokens_out,
        duration_ms: ctx.started_at.elapsed().as_millis() as u64,
        status,
        error,
    });
}

fn http_status_from_error(e: &crabllm_core::Error) -> u16 {
    match e {
        crabllm_core::Error::Provider { status, .. } => *status,
        crabllm_core::Error::Timeout => 504,
        _ => 500,
    }
}

pub(crate) fn emit_usage_error<S: Storage, P: Provider>(
    state: &AppState<S, P>,
    ctx: &RequestContext,
    endpoint: &'static str,
    e: &crabllm_core::Error,
) {
    emit_usage(
        state,
        ctx,
        endpoint,
        0,
        0,
        http_status_from_error(e),
        Some(e.to_string()),
    );
}

/// POST /v1/chat/completions
pub async fn chat_completions<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(key_name): Extension<KeyName>,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
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
            match try_stream_with_retries(deployment, &request).await {
                Ok(stream) => {
                    let extensions = state.extensions.clone();
                    let ctx = Arc::new(ctx);
                    let errored = Arc::new(AtomicBool::new(false));
                    // `Ordering::Relaxed` is sufficient because the stream
                    // is polled by a single task — every store in the
                    // per-chunk future is sequenced-before the `done`
                    // terminator's load via tokio's normal single-task
                    // poll ordering. No cross-thread visibility problem
                    // exists; the atomics only provide interior
                    // mutability across closure clones.
                    let tokens_in = Arc::new(AtomicU32::new(0));
                    let tokens_out = Arc::new(AtomicU32::new(0));
                    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

                    let ctx_done = ctx.clone();
                    let errored_done = errored.clone();
                    let tokens_in_done = tokens_in.clone();
                    let tokens_out_done = tokens_out.clone();
                    let first_error_done = first_error.clone();
                    let state_done = state.clone();

                    let observed = stream.then(move |result| {
                        let extensions = extensions.clone();
                        let ctx = ctx.clone();
                        let errored = errored.clone();
                        let tokens_in = tokens_in.clone();
                        let tokens_out = tokens_out.clone();
                        let first_error = first_error.clone();
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
                                        tokens_in.store(usage.prompt_tokens, Ordering::Relaxed);
                                        tokens_out
                                            .store(usage.completion_tokens, Ordering::Relaxed);
                                    }
                                    for ext in extensions.iter() {
                                        ext.on_chunk(&ctx, chunk).await;
                                    }
                                }
                                Err(error) => {
                                    errored.store(true, Ordering::Relaxed);
                                    {
                                        let mut slot = first_error.lock().unwrap();
                                        if slot.is_none() {
                                            *slot = Some(error.to_string());
                                        }
                                    }
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

                    // Record duration + emit usage once when the stream
                    // terminates. Status codes:
                    //   - 200: clean stream, no errors
                    //   - 0:   mid-stream failure after headers went out.
                    //          The real wire status was 200 (headers
                    //          shipped before the break), but reporting
                    //          200 again here would let consumers miss
                    //          the failure without also inspecting the
                    //          `error` field. 0 is a clear sentinel
                    //          meaning "not a real HTTP response".
                    let done = futures::stream::once(async move {
                        let errored = errored_done.load(Ordering::Relaxed);
                        record_duration(&ctx_done, if errored { "5xx" } else { "2xx" });
                        let error = first_error_done.lock().unwrap().take();
                        let status = if errored { 0 } else { 200 };
                        emit_usage(
                            &state_done,
                            &ctx_done,
                            "chat.completions",
                            tokens_in_done.load(Ordering::Relaxed),
                            tokens_out_done.load(Ordering::Relaxed),
                            status,
                            error,
                        );
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
        emit_usage_error(&state, &ctx, "chat.completions", &e);
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
            match try_chat_with_retries(deployment, &request).await {
                Ok(resp) => {
                    let (pt, ct) = resp
                        .usage
                        .as_ref()
                        .map(|u| (u.prompt_tokens, u.completion_tokens))
                        .unwrap_or((0, 0));
                    if pt > 0 || ct > 0 {
                        record_tokens(&ctx, pt, ct);
                    }
                    record_duration(&ctx, "2xx");
                    emit_usage(&state, &ctx, "chat.completions", pt, ct, 200, None);
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
        emit_usage_error(&state, &ctx, "chat.completions", &e);
        error_response(e)
    }
}

/// POST /v1/embeddings
pub async fn embeddings<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<EmbeddingRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
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
        match try_embedding_with_retries(deployment, &request).await {
            Ok(resp) => {
                record_duration(&ctx, "2xx");
                emit_usage(
                    &state,
                    &ctx,
                    "embeddings",
                    resp.usage.prompt_tokens,
                    0,
                    200,
                    None,
                );
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
    emit_usage_error(&state, &ctx, "embeddings", &e);
    error_response(e)
}

/// GET /v1/models
///
/// Serves OpenAI-shaped model list by default. When the request carries an
/// Anthropic-flavored auth header (`x-api-key` without `Authorization: Bearer`)
/// or the `anthropic-version` header, returns Anthropic's model-list shape
/// instead, so the official Anthropic SDKs see the response format they expect.
pub async fn models<S, P>(State(state): State<AppState<S, P>>, headers: HeaderMap) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let names: Vec<String> = state
        .registry
        .model_names()
        .map(|n| n.to_string())
        .collect();

    if is_anthropic_client(&headers) {
        let data: Vec<serde_json::Value> = names
            .into_iter()
            .map(|id| {
                serde_json::json!({
                    "type": "model",
                    "id": id.clone(),
                    "display_name": id,
                    "created_at": "1970-01-01T00:00:00Z",
                })
            })
            .collect();
        return Json(serde_json::json!({
            "data": data,
            "has_more": false,
            "first_id": null,
            "last_id": null,
        }))
        .into_response();
    }

    let data: Vec<Model> = names
        .into_iter()
        .map(|name| Model {
            id: name,
            object: "model".to_string(),
            created: 0,
            owned_by: "crabllm".to_string(),
        })
        .collect();

    Json(ModelList {
        object: "list".to_string(),
        data,
    })
    .into_response()
}

fn is_anthropic_client(headers: &HeaderMap) -> bool {
    if headers.contains_key("anthropic-version") {
        return true;
    }
    // `x-api-key` without a Bearer Authorization header is the Anthropic SDK's
    // default auth shape. OpenAI SDKs always send Authorization: Bearer.
    headers.contains_key("x-api-key")
        && !headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|h| h.starts_with("Bearer "))
            .unwrap_or(false)
}

/// POST /v1/images/generations
pub async fn image_generations<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<ImageRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
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
            deployment.provider.image_generation(&request),
        )
        .await
        {
            Ok((bytes, content_type)) => {
                record_duration(&ctx, "2xx");
                emit_usage(&state, &ctx, "images.generations", 0, 0, 200, None);
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
    emit_usage_error(&state, &ctx, "images.generations", &e);
    error_response(e)
}

/// POST /v1/audio/speech
pub async fn audio_speech<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<AudioSpeechRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
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
            deployment.provider.audio_speech(&request),
        )
        .await
        {
            Ok((bytes, content_type)) => {
                record_duration(&ctx, "2xx");
                emit_usage(&state, &ctx, "audio.speech", 0, 0, 200, None);
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
    emit_usage_error(&state, &ctx, "audio.speech", &e);
    error_response(e)
}

/// POST /v1/audio/transcriptions
pub async fn audio_transcriptions<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(key_name): Extension<KeyName>,
    mut multipart: Multipart,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
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
        fields.push(MultipartField {
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

    // Fallback only (no retry loop) — provider impls rebuild a fresh
    // multipart form per call from the buffered field slice.
    let mut last_err = None;
    for deployment in &deployments {
        match with_timeout(
            deployment.timeout,
            deployment.provider.audio_transcription(&model, &fields),
        )
        .await
        {
            Ok((bytes, content_type)) => {
                record_duration(&ctx, "2xx");
                emit_usage(&state, &ctx, "audio.transcriptions", 0, 0, 200, None);
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
    emit_usage_error(&state, &ctx, "audio.transcriptions", &e);
    error_response(e)
}

/// Call a provider with a timeout, converting elapsed to Error::Timeout.
/// A zero duration disables the timeout.
pub(crate) async fn with_timeout<F, T>(timeout: Duration, fut: F) -> Result<T, crabllm_core::Error>
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
pub(crate) async fn try_chat_with_retries<P: Provider>(
    deployment: &Deployment<P>,
    request: &ChatCompletionRequest,
) -> Result<crabllm_core::ChatCompletionResponse, crabllm_core::Error> {
    let mut last_err;
    match with_timeout(
        deployment.timeout,
        deployment.provider.chat_completion(request),
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
            deployment.provider.chat_completion(request),
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
pub(crate) async fn try_stream_with_retries<P: Provider>(
    deployment: &Deployment<P>,
    request: &ChatCompletionRequest,
) -> Result<BoxStream<'static, Result<ChatCompletionChunk, crabllm_core::Error>>, crabllm_core::Error>
{
    let mut last_err;
    match with_timeout(
        deployment.timeout,
        deployment.provider.chat_completion_stream(request),
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
            deployment.provider.chat_completion_stream(request),
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
async fn try_embedding_with_retries<P: Provider>(
    deployment: &Deployment<P>,
    request: &EmbeddingRequest,
) -> Result<crabllm_core::EmbeddingResponse, crabllm_core::Error> {
    let mut last_err;
    match with_timeout(deployment.timeout, deployment.provider.embedding(request)).await {
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
        match with_timeout(deployment.timeout, deployment.provider.embedding(request)).await {
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
pub(crate) fn error_response(e: crabllm_core::Error) -> Response {
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

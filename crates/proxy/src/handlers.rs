use crate::{AppState, auth::Principal, state::UsageEvent};
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
    ApiError, AudioSpeechRequest, EmbeddingRequest, ImageRequest, Model, ModelList, MultipartField,
    Provider, RequestContext, Storage,
};
use crabllm_provider::Deployment;
use futures::StreamExt;
use parking_lot::Mutex;
use rand::Rng;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
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
    match e.http_status() {
        429 => "429",
        400..=499 => "4xx",
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

pub(crate) struct RequestOutcome {
    pub usage: crabllm_core::Usage,
    pub status: u16,
    pub error: Option<String>,
}

impl RequestOutcome {
    pub fn ok(usage: crabllm_core::Usage) -> Self {
        Self {
            usage,
            status: 200,
            error: None,
        }
    }
}

pub(crate) fn emit_usage<S: Storage, P: Provider>(
    state: &AppState<S, P>,
    ctx: &RequestContext,
    endpoint: &'static str,
    outcome: RequestOutcome,
) {
    let Some(tx) = state.usage_events.as_ref() else {
        return;
    };
    let _ = tx.send(UsageEvent {
        timestamp: SystemTime::now(),
        request_id: ctx.request_id.clone(),
        principal: ctx.principal.clone(),
        model: ctx.model.clone(),
        provider: ctx.provider.clone(),
        endpoint,
        usage: outcome.usage,
        duration_ms: ctx.started_at.elapsed().as_millis() as u64,
        status: outcome.status,
        error: outcome.error,
    });
}

/// GET /v1/usage — user-facing, auto-scoped to the caller's key.
pub(crate) async fn usage<S: Storage + 'static, P: Provider + 'static>(
    State(state): State<AppState<S, P>>,
    Extension(Principal(principal)): Extension<Principal>,
    query: axum::extract::Query<crate::ext::usage::UserUsageQuery>,
) -> Json<Vec<crate::ext::usage::UsageEntry>> {
    let name = principal.as_deref().unwrap_or("__global");
    crate::ext::usage::query_usage(
        state.storage.as_ref() as &dyn Storage,
        Some(name),
        query.model.as_deref(),
    )
    .await
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
        RequestOutcome {
            usage: crabllm_core::Usage::default(),
            status: e.http_status(),
            error: Some(e.to_string()),
        },
    );
}

fn read_error_response(e: crate::body::ReadError) -> Response {
    match e {
        crate::body::ReadError::Io(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(msg, "server_error")),
        )
            .into_response(),
        crate::body::ReadError::InvalidJson(msg) => (
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(msg, "invalid_request_error")),
        )
            .into_response(),
    }
}

/// POST /v1/chat/completions
pub async fn chat_completions<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(principal): Extension<Principal>,
    body: axum::body::Body,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let body = match crate::body::RequestBody::read(body).await {
        Ok(b) => b,
        Err(e) => return read_error_response(e),
    };
    let registry = state.registry();
    let model = registry.resolve(&body.model).to_string();
    let Some(deployments) = registry.dispatch_list(&model) else {
        return (
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("model '{model}' not found"),
                "invalid_request_error",
            )),
        )
            .into_response();
    };
    let is_stream = body.is_stream;

    if is_stream {
        return handle_raw_stream_passthrough(
            &state,
            principal,
            &model,
            &deployments,
            body.into_stream(),
        )
        .await;
    }

    let raw_body = match body.into_bytes().await {
        Ok(b) => b,
        Err(e) => return read_error_response(e),
    };

    handle_raw_proxy(&state, principal, &model, &deployments, raw_body).await
}

/// Non-streaming raw byte proxy for OpenAI-compatible providers.
/// Forwards the request body without deserialization and returns the
/// response bytes directly. Extensions receive raw bytes.
async fn handle_raw_proxy<S: Storage, P: Provider>(
    state: &AppState<S, P>,
    principal: Principal,
    model: &str,
    deployments: &[&Deployment<P>],
    raw_body: axum::body::Bytes,
) -> Response {
    let registry = state.registry();
    let provider_name = registry
        .provider_name(model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.to_string(),
        provider: provider_name,
        principal: principal.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
        }
    }

    for ext in state.extensions.iter() {
        if let Some(cached) = ext.on_cache_lookup(&raw_body).await {
            return (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                cached,
            )
                .into_response();
        }
    }

    let mut last_err = None;
    for deployment in deployments {
        match with_timeout(
            deployment.timeout,
            deployment
                .provider
                .chat_completion_raw(model, raw_body.clone()),
        )
        .await
        {
            Ok(resp_bytes) => {
                let usage = crabllm_core::Usage::from(resp_bytes.as_ref());
                if usage.prompt_tokens() > 0 || usage.completion_tokens() > 0 {
                    record_tokens(&ctx, usage.prompt_tokens(), usage.completion_tokens());
                }
                record_duration(&ctx, "2xx");
                emit_usage(state, &ctx, "chat.completions", RequestOutcome::ok(usage));
                for ext in state.extensions.iter() {
                    ext.on_response(&ctx, &raw_body, &resp_bytes).await;
                }
                return (
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    resp_bytes,
                )
                    .into_response();
            }
            Err(e) => {
                if !e.is_transient() {
                    for ext in state.extensions.iter() {
                        ext.on_error(&ctx, &e).await;
                    }
                    record_duration(&ctx, error_status(&e));
                    emit_usage_error(state, &ctx, "chat.completions", &e);
                    return error_response(e);
                }
                last_err = Some(e);
            }
        }
    }

    let e = last_err
        .unwrap_or_else(|| crabllm_core::Error::Routing("no providers available".to_string()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, "chat.completions", &e);
    error_response(e)
}

/// Parse an OpenAI SSE byte stream into raw `data:` payload strings.
/// Terminates on `[DONE]`.
fn openai_raw_sse(
    byte_stream: crabllm_core::ByteStream,
) -> impl futures::Stream<Item = Result<String, crabllm_core::Error>> {
    use bytes::{Buf, BytesMut};
    futures::stream::unfold(
        (byte_stream, BytesMut::new()),
        |(mut bytes, mut buf)| async move {
            loop {
                if let Some(newline_pos) = buf.iter().position(|&b| b == b'\n') {
                    let mut line_end = newline_pos;
                    if line_end > 0 && buf[line_end - 1] == b'\r' {
                        line_end -= 1;
                    }
                    let line = &buf[..line_end];

                    if let Some(rest) = line.strip_prefix(b"data: ")
                        && let Ok(s) = std::str::from_utf8(rest)
                    {
                        let data = s.trim().to_string();
                        buf.advance(newline_pos + 1);
                        if data == "[DONE]" {
                            return None;
                        }
                        return Some((Ok(data), (bytes, buf)));
                    }

                    buf.advance(newline_pos + 1);
                    continue;
                }

                match bytes.next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(e)) => {
                        return Some((
                            Err(crabllm_core::Error::Network(e.to_string())),
                            (bytes, buf),
                        ));
                    }
                    None => return None,
                }
            }
        },
    )
}

/// Streaming passthrough for OpenAI-compatible providers.
/// Body is streamed from client → upstream without full buffering.
/// Single attempt — no retries (body stream is consumed).
async fn handle_raw_stream_passthrough<S: Storage + 'static, P: Provider + 'static>(
    state: &AppState<S, P>,
    principal: Principal,
    model: &str,
    deployments: &[&Deployment<P>],
    body_stream: crabllm_core::ByteStream,
) -> Response {
    let registry = state.registry();
    let provider_name = registry
        .provider_name(model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.to_string(),
        provider: provider_name,
        principal: principal.0,
        is_stream: true,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
        }
    }

    let deployment = match deployments.iter().find(|d| d.provider.is_openai_compat()) {
        Some(d) => d,
        None => {
            let e = crabllm_core::Error::Routing("no compatible providers available".to_string());
            record_duration(&ctx, "5xx");
            emit_usage_error(state, &ctx, "chat.completions", &e);
            return error_response(e);
        }
    };

    let byte_stream = match with_timeout(
        deployment.timeout,
        deployment
            .provider
            .chat_completion_stream_passthrough(model, body_stream),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            for ext in state.extensions.iter() {
                ext.on_error(&ctx, &e).await;
            }
            record_duration(&ctx, "5xx");
            emit_usage_error(state, &ctx, "chat.completions", &e);
            return error_response(e);
        }
    };

    let extensions = state.extensions.clone();
    let ctx = Arc::new(ctx);
    let errored = Arc::new(AtomicBool::new(false));
    let usage: Arc<Mutex<crabllm_core::Usage>> =
        Arc::new(Mutex::new(crabllm_core::Usage::default()));
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let ctx_done = ctx.clone();
    let errored_done = errored.clone();
    let usage_done = usage.clone();
    let first_error_done = first_error.clone();
    let state_done = state.clone();

    let events = openai_raw_sse(byte_stream);
    let sse_stream = events.map(move |result| {
        let extensions = extensions.clone();
        let ctx = ctx.clone();
        let errored = errored.clone();
        let usage = usage.clone();
        let first_error = first_error.clone();
        async move {
            match result {
                Ok(data) => {
                    let peeked = crabllm_core::Usage::from(data.as_bytes());
                    if peeked.total_tokens() > 0 {
                        record_tokens(&ctx, peeked.prompt_tokens(), peeked.completion_tokens());
                        *usage.lock() = peeked;
                    }
                    for ext in extensions.iter() {
                        ext.on_chunk(&ctx, data.as_bytes()).await;
                    }
                    Ok::<_, std::convert::Infallible>(Event::default().data(data))
                }
                Err(error) => {
                    errored.store(true, Ordering::Relaxed);
                    {
                        let mut slot = first_error.lock();
                        if slot.is_none() {
                            *slot = Some(error.to_string());
                        }
                    }
                    for ext in extensions.iter() {
                        ext.on_error(&ctx, &error).await;
                    }
                    let json = crabllm_core::json::to_string(&ApiError::new(
                        error.to_string(),
                        "server_error",
                    ))
                    .unwrap_or_default();
                    Ok(Event::default().data(json))
                }
            }
        }
    });
    let buffered = sse_stream.buffered(1);

    let done = futures::stream::once(async move {
        let errored = errored_done.load(Ordering::Relaxed);
        record_duration(&ctx_done, if errored { "5xx" } else { "2xx" });
        let error = first_error_done.lock().take();
        let status = if errored { 0 } else { 200 };
        let usage = usage_done.lock().clone();
        emit_usage(
            &state_done,
            &ctx_done,
            "chat.completions",
            RequestOutcome {
                usage,
                status,
                error,
            },
        );
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let full_stream = buffered.chain(done);

    Sse::new(full_stream)
        .keep_alive(axum::response::sse::KeepAlive::new())
        .into_response()
}

/// POST /v1/embeddings
pub async fn embeddings<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(principal): Extension<Principal>,
    Json(request): Json<EmbeddingRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let registry = state.registry();
    let model = registry.resolve(&request.model).to_string();
    let deployments = match registry.dispatch_list(&model) {
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

    let provider_name = registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        principal: principal.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    // Run on_request hooks — short-circuit on first error.
    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
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
                    RequestOutcome::ok(crabllm_core::Usage {
                        input_tokens: resp.usage.prompt_tokens,
                        ..Default::default()
                    }),
                );
                return Json(resp).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Routing("no providers available".into()));
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
    let registry = state.registry();
    let names: Vec<String> = registry.model_names().map(|n| n.to_string()).collect();

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
        .map(|name| {
            let info = state.config.models.get(name.as_str());
            let dialects = registry.model_dialects(&name);
            Model {
                id: name,
                object: "model".to_string(),
                created: 0,
                owned_by: "crabllm".to_string(),
                context_length: info.and_then(|i| i.context_length),
                pricing: info.and_then(|i| i.pricing.clone()),
                vision: info.and_then(|i| i.vision),
                dialects,
            }
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
    Extension(principal): Extension<Principal>,
    Json(request): Json<ImageRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let registry = state.registry();
    let model = registry.resolve(&request.model).to_string();
    let deployments = match registry.dispatch_list(&model) {
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

    let provider_name = registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        principal: principal.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
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
                emit_usage(
                    &state,
                    &ctx,
                    "images.generations",
                    RequestOutcome::ok(crabllm_core::Usage::default()),
                );
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Routing("no providers available".into()));
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
    Extension(principal): Extension<Principal>,
    Json(request): Json<AudioSpeechRequest>,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let registry = state.registry();
    let model = registry.resolve(&request.model).to_string();
    let deployments = match registry.dispatch_list(&model) {
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

    let provider_name = registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        principal: principal.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
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
                emit_usage(
                    &state,
                    &ctx,
                    "audio.speech",
                    RequestOutcome::ok(crabllm_core::Usage::default()),
                );
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Routing("no providers available".into()));
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
    Extension(principal): Extension<Principal>,
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

    let registry = state.registry();
    let model = match model_value {
        Some(m) => registry.resolve(&m).to_string(),
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

    let deployments = match registry.dispatch_list(&model) {
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

    let provider_name = registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        principal: principal.0,
        is_stream: false,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
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
                emit_usage(
                    &state,
                    &ctx,
                    "audio.transcriptions",
                    RequestOutcome::ok(crabllm_core::Usage::default()),
                );
                return ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response();
            }
            Err(e) => last_err = Some(e),
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Routing("no providers available".into()));
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

/// Pick the retry sleep duration. On 429, honor the upstream Retry-After
/// (floored at the exponential backoff so we don't under-wait). For all
/// other transient errors, use the jittered backoff as-is.
fn retry_sleep(err: &crabllm_core::Error, backoff: Duration) -> Duration {
    match err.retry_after() {
        Some(ra) => ra.max(backoff),
        None => backoff,
    }
}

/// Retry a raw Anthropic streaming request on a single deployment.
pub(crate) async fn try_anthropic_stream_with_retries<P: Provider>(
    deployment: &Deployment<P>,
    raw_body: bytes::Bytes,
) -> Result<crabllm_core::ByteStream, crabllm_core::Error> {
    let started = Instant::now();
    let mut last_err;
    match with_timeout(
        deployment.timeout,
        deployment
            .provider
            .anthropic_messages_stream_raw(raw_body.clone()),
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
        let sleep = retry_sleep(&last_err, jittered(backoff));
        if started.elapsed() + sleep > deployment.retry_deadline {
            break;
        }
        tokio::time::sleep(sleep).await;
        backoff *= 2;
        match with_timeout(
            deployment.timeout,
            deployment
                .provider
                .anthropic_messages_stream_raw(raw_body.clone()),
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
    let started = Instant::now();
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
        let sleep = retry_sleep(&last_err, jittered(backoff));
        if started.elapsed() + sleep > deployment.retry_deadline {
            break;
        }
        tokio::time::sleep(sleep).await;
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

/// Map a provider Error to an HTTP error response. Status and `type` come from
/// the error variant ([`Error::http_status`]/[`Error::kind`]); the message is
/// the upstream's own body for `Provider`, the error's `Display` otherwise.
pub(crate) fn error_response(e: crabllm_core::Error) -> Response {
    let status = StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::BAD_GATEWAY);
    let message = match &e {
        crabllm_core::Error::Provider { body, .. } => body.clone(),
        _ => e.to_string(),
    };
    (status, Json(ApiError::new(message, e.kind()))).into_response()
}

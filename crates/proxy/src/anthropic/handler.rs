//! HTTP handler for `POST /v1/messages` (Anthropic-compatible).
//!
//! For Anthropic-compatible upstreams (Anthropic, DeepSeek), requests are
//! forwarded as raw Anthropic-format bytes — no format translation. For
//! other upstreams, the request is translated to the internal OpenAI format,
//! dispatched, and the response translated back.

use crate::{
    AppState,
    anthropic::{from_chat_completion, to_anthropic_sse, to_chat_completion},
    auth::Principal,
    handlers::{
        RequestOutcome, emit_usage, emit_usage_error, error_response, error_status,
        record_duration, record_tokens, try_anthropic_stream_with_retries, try_chat_with_retries,
        try_stream_with_retries,
    },
};
use axum::{
    Extension, Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use bytes::{Buf, BytesMut};
use crabllm_core::{ApiError, Provider, RequestContext, Storage};
use futures::StreamExt;
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

const ENDPOINT: &str = "messages";

/// Lightweight peek for routing the Anthropic request.
#[derive(serde::Deserialize)]
struct AnthropicPeek {
    model: String,
    #[serde(default)]
    stream: Option<bool>,
}

#[allow(clippy::result_large_err)]
fn deserialize_request(raw_body: &[u8]) -> Result<crabllm_core::ChatCompletionRequest, Response> {
    let anthropic_req: crabllm_core::AnthropicRequest = crabllm_core::json::from_slice(raw_body)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(e.to_string(), "invalid_request_error")),
            )
                .into_response()
        })?;
    Ok(to_chat_completion(anthropic_req))
}

/// POST /v1/messages
pub async fn messages<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(principal): Extension<Principal>,
    raw_body: axum::body::Bytes,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let peek: AnthropicPeek = match crabllm_core::json::from_slice(&raw_body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(e.to_string(), "invalid_request_error")),
            )
                .into_response();
        }
    };
    let is_stream = peek.stream == Some(true);
    let registry = state.registry();
    let model = registry.resolve(&peek.model).to_string();
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

    // Raw byte proxy: non-streaming, Anthropic-compatible upstream.
    // Extensions receive raw bytes and deserialize only what they need.
    if !is_stream && deployments.iter().all(|d| d.provider.is_anthropic_compat()) {
        return handle_raw_anthropic(&state, principal, &model, &deployments, raw_body).await;
    }

    let provider_name = registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        principal: principal.0,
        is_stream,
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

    if is_stream {
        let mut last_err = None;
        for deployment in &deployments {
            // Raw Anthropic streaming for compatible providers — no format
            // translation, the upstream speaks Anthropic SSE natively.
            if deployment.provider.is_anthropic_compat() {
                match try_anthropic_stream_with_retries(deployment, raw_body.clone()).await {
                    Ok(byte_stream) => {
                        return raw_anthropic_stream_response(byte_stream, &state, ctx);
                    }
                    Err(e) => {
                        last_err = Some(e);
                        continue;
                    }
                }
            }

            // Translated streaming for non-compatible providers — deserialize
            // on first use so anthropic-compat paths never pay this cost.
            let mut request = match deserialize_request(&raw_body) {
                Ok(r) => r,
                Err(resp) => return resp,
            };
            request
                .extra
                .entry("stream_options".to_string())
                .or_insert(serde_json::json!({ "include_usage": true }));
            match try_stream_with_retries(deployment, &request).await {
                Ok(stream) => {
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

                    let observed = stream.then(move |result| {
                        let extensions = extensions.clone();
                        let ctx = ctx.clone();
                        let errored = errored.clone();
                        let usage = usage.clone();
                        let first_error = first_error.clone();
                        async move {
                            match &result {
                                Ok(chunk) => {
                                    if let Some(ref wire) = chunk.usage {
                                        let canonical = crabllm_core::Usage::from(wire);
                                        record_tokens(
                                            &ctx,
                                            canonical.prompt_tokens(),
                                            canonical.completion_tokens(),
                                        );
                                        *usage.lock() = canonical;
                                    }
                                    if !extensions.is_empty() {
                                        let raw = crabllm_core::json::to_vec(chunk)
                                            .unwrap_or_default();
                                        for ext in extensions.iter() {
                                            ext.on_chunk(&ctx, &raw).await;
                                        }
                                    }
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
                                        ext.on_error(&ctx, error).await;
                                    }
                                }
                            }
                            result
                        }
                    });

                    let anthropic_events = to_anthropic_sse(Box::pin(observed));

                    let sse_stream = anthropic_events.map(|result| match result {
                        Ok(event) => {
                            let name = event.event_name();
                            let json = crabllm_core::json::to_string(&event).unwrap_or_default();
                            Ok::<_, std::convert::Infallible>(
                                Event::default().event(name).data(json),
                            )
                        }
                        Err(e) => {
                            let json = crabllm_core::json::to_string(&serde_json::json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": e.to_string(),
                                },
                            }))
                            .unwrap_or_default();
                            Ok(Event::default().event("error").data(json))
                        }
                    });

                    let finalized = futures::stream::unfold(
                        (
                            Box::pin(sse_stream),
                            Some((
                                state.clone(),
                                ctx_done,
                                usage_done,
                                errored_done,
                                first_error_done,
                            )),
                        ),
                        |(mut inner, mut slot)| async move {
                            match inner.next().await {
                                Some(item) => Some((item, (inner, slot))),
                                None => {
                                    if let Some((state, ctx, u, er, fe)) = slot.take() {
                                        let errored = er.load(Ordering::Relaxed);
                                        record_duration(&ctx, if errored { "5xx" } else { "2xx" });
                                        let error = fe.lock().take();
                                        let status = if errored { 0 } else { 200 };
                                        let usage = u.lock().clone();
                                        emit_usage(
                                            &state,
                                            &ctx,
                                            ENDPOINT,
                                            RequestOutcome {
                                                usage,
                                                status,
                                                error,
                                            },
                                        );
                                    }
                                    None
                                }
                            }
                        },
                    );

                    return Sse::new(finalized)
                        .keep_alive(axum::response::sse::KeepAlive::new())
                        .into_response();
                }
                Err(e) => last_err = Some(e),
            }
        }

        let e = last_err
            .unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
        for ext in state.extensions.iter() {
            ext.on_error(&ctx, &e).await;
        }
        record_duration(&ctx, "5xx");
        emit_usage_error(&state, &ctx, ENDPOINT, &e);
        return error_response(e);
    }

    // Non-streaming non-compat: deserialization needed for format translation.
    // Extensions still receive raw bytes.
    for ext in state.extensions.iter() {
        if let Some(cached) = ext.on_cache_lookup(&raw_body).await {
            return (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                cached,
            )
                .into_response();
        }
    }

    let request = match deserialize_request(&raw_body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let mut last_err = None;
    for deployment in &deployments {
        match try_chat_with_retries(deployment, &request).await {
            Ok(resp) => {
                match from_chat_completion(resp) {
                    Ok(anthropic) => {
                        let resp_bytes =
                            crabllm_core::json::to_vec(&anthropic).unwrap_or_default();
                        let usage = crabllm_core::Usage::from(resp_bytes.as_slice());
                        if usage.prompt_tokens() > 0 || usage.completion_tokens() > 0 {
                            record_tokens(&ctx, usage.prompt_tokens(), usage.completion_tokens());
                        }
                        record_duration(&ctx, "2xx");
                        emit_usage(&state, &ctx, ENDPOINT, RequestOutcome::ok(usage));
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
                        for ext in state.extensions.iter() {
                            ext.on_error(&ctx, &e).await;
                        }
                        record_duration(&ctx, error_status(&e));
                        emit_usage_error(&state, &ctx, ENDPOINT, &e);
                        return error_response(e);
                    }
                }
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
    emit_usage_error(&state, &ctx, ENDPOINT, &e);
    error_response(e)
}

/// Non-streaming raw byte proxy for Anthropic-compatible providers.
/// Extensions receive raw bytes and deserialize only what they need.
async fn handle_raw_anthropic<S: Storage, P: Provider>(
    state: &AppState<S, P>,
    principal: Principal,
    model: &str,
    deployments: &[&crabllm_provider::Deployment<P>],
    raw_body: axum::body::Bytes,
) -> Response {
    use crate::handlers::with_timeout;

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
            return (
                StatusCode::from_u16(ext_err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(ext_err.body),
            )
                .into_response();
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
            deployment.provider.anthropic_messages_raw(raw_body.clone()),
        )
        .await
        {
            Ok(resp_bytes) => {
                let usage = crabllm_core::Usage::from(resp_bytes.as_ref());
                if usage.prompt_tokens() > 0 || usage.completion_tokens() > 0 {
                    record_tokens(&ctx, usage.prompt_tokens(), usage.completion_tokens());
                }
                record_duration(&ctx, "2xx");
                emit_usage(state, &ctx, ENDPOINT, RequestOutcome::ok(usage));
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
                    emit_usage_error(state, &ctx, ENDPOINT, &e);
                    return error_response(e);
                }
                last_err = Some(e);
            }
        }
    }

    let e = last_err
        .unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".to_string()));
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, ENDPOINT, &e);
    error_response(e)
}

/// Streaming raw byte proxy for Anthropic-compatible providers.
///
/// Forwards upstream Anthropic SSE events directly to the client, parsing
/// just enough to extract usage tokens for metrics.
fn raw_anthropic_stream_response<S: Storage + 'static, P: Provider + 'static>(
    byte_stream: crabllm_core::ByteStream,
    state: &AppState<S, P>,
    ctx: RequestContext,
) -> Response {
    let ctx = Arc::new(ctx);
    let usage: Arc<Mutex<crabllm_core::Usage>> =
        Arc::new(Mutex::new(crabllm_core::Usage::default()));
    let errored = Arc::new(AtomicBool::new(false));
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let ctx_c = ctx.clone();
    let usage_c = usage.clone();
    let errored_c = errored.clone();
    let first_error_c = first_error.clone();

    let events = anthropic_raw_sse(byte_stream);

    let sse_stream = events.map(move |result| match result {
        Ok((event_name, data)) => {
            peek_anthropic_usage(&event_name, &data, &usage_c);
            let snapshot = usage_c.lock().clone();
            if snapshot.prompt_tokens() > 0 || snapshot.completion_tokens() > 0 {
                record_tokens(&ctx_c, snapshot.prompt_tokens(), snapshot.completion_tokens());
            }
            Ok::<_, std::convert::Infallible>(Event::default().event(event_name).data(data))
        }
        Err(e) => {
            errored_c.store(true, Ordering::Relaxed);
            {
                let mut slot = first_error_c.lock();
                if slot.is_none() {
                    *slot = Some(e.to_string());
                }
            }
            let json = crabllm_core::json::to_string(&serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": e.to_string(),
                },
            }))
            .unwrap_or_default();
            Ok(Event::default().event("error").data(json))
        }
    });

    let state = state.clone();
    let finalized = futures::stream::unfold(
        (
            Box::pin(sse_stream),
            Some((state, ctx, usage, errored, first_error)),
        ),
        |(mut inner, mut slot)| async move {
            match inner.next().await {
                Some(item) => Some((item, (inner, slot))),
                None => {
                    if let Some((state, ctx, u, er, fe)) = slot.take() {
                        let errored = er.load(Ordering::Relaxed);
                        record_duration(&ctx, if errored { "5xx" } else { "2xx" });
                        let error = fe.lock().take();
                        let status = if errored { 0 } else { 200 };
                        let usage = u.lock().clone();
                        emit_usage(
                            &state,
                            &ctx,
                            ENDPOINT,
                            RequestOutcome {
                                usage,
                                status,
                                error,
                            },
                        );
                    }
                    None
                }
            }
        },
    );

    Sse::new(finalized)
        .keep_alive(axum::response::sse::KeepAlive::new())
        .into_response()
}

/// Parse a raw SSE byte stream into `(event_name, data)` pairs.
fn anthropic_raw_sse(
    byte_stream: crabllm_core::ByteStream,
) -> impl futures::Stream<Item = Result<(String, String), crabllm_core::Error>> {
    futures::stream::unfold(
        (byte_stream, BytesMut::new(), None::<String>, None::<String>),
        |(mut bytes, mut buf, mut event_name, mut data)| async move {
            use futures::StreamExt;
            loop {
                if let Some(newline_pos) = buf.iter().position(|&b| b == b'\n') {
                    let mut line_end = newline_pos;
                    if line_end > 0 && buf[line_end - 1] == b'\r' {
                        line_end -= 1;
                    }
                    let line = &buf[..line_end];

                    if line.is_empty() {
                        buf.advance(newline_pos + 1);
                        if let (Some(name), Some(d)) = (event_name.take(), data.take()) {
                            return Some((Ok((name, d)), (bytes, buf, None, None)));
                        }
                        continue;
                    }

                    if let Some(rest) = line.strip_prefix(b"event: ")
                        && let Ok(s) = std::str::from_utf8(rest)
                    {
                        event_name = Some(s.trim().to_string());
                    } else if let Some(rest) = line.strip_prefix(b"data: ")
                        && let Ok(s) = std::str::from_utf8(rest)
                    {
                        data = Some(s.trim().to_string());
                    }

                    buf.advance(newline_pos + 1);
                    continue;
                }

                match bytes.next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(e)) => {
                        return Some((
                            Err(crabllm_core::Error::Internal(format!("stream error: {e}"))),
                            (bytes, buf, event_name, data),
                        ));
                    }
                    None => return None,
                }
            }
        },
    )
}

/// Extract canonical usage axes from raw Anthropic SSE event data. Reads
/// `message_start.usage` (input + cache_read + cache_create) and
/// `message_delta.usage` (output_tokens) into the shared snapshot. Subsequent
/// events overwrite — Anthropic's stream usage is cumulative-to-this-point.
fn peek_anthropic_usage(event_name: &str, data: &str, usage: &Mutex<crabllm_core::Usage>) {
    let val: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return,
    };

    match event_name {
        "message_start" => {
            if let Some(u) = val.pointer("/message/usage") {
                let mut snap = usage.lock();
                snap.input_tokens = u
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                snap.cache_read_tokens = u
                    .get("cache_read_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                snap.cache_write_tokens = u
                    .get("cache_creation_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
            }
        }
        "message_delta" => {
            if let Some(u) = val.get("usage")
                && let Some(n) = u.get("output_tokens").and_then(|v| v.as_u64())
            {
                usage.lock().output_tokens = n as u32;
            }
        }
        _ => {}
    }
}

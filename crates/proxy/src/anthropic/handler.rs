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
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU32, Ordering},
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

    // Raw byte proxy: non-streaming, no extensions, Anthropic-compatible upstream.
    if !is_stream
        && state.extensions.is_empty()
        && deployments.iter().all(|d| d.provider.is_anthropic_compat())
    {
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
                    let tokens_in = Arc::new(AtomicU32::new(0));
                    let tokens_out = Arc::new(AtomicU32::new(0));
                    let cache_hit = Arc::new(AtomicU32::new(0));
                    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

                    let ctx_done = ctx.clone();
                    let errored_done = errored.clone();
                    let tokens_in_done = tokens_in.clone();
                    let tokens_out_done = tokens_out.clone();
                    let cache_hit_done = cache_hit.clone();
                    let first_error_done = first_error.clone();

                    let observed = stream.then(move |result| {
                        let extensions = extensions.clone();
                        let ctx = ctx.clone();
                        let errored = errored.clone();
                        let tokens_in = tokens_in.clone();
                        let tokens_out = tokens_out.clone();
                        let cache_hit = cache_hit.clone();
                        let first_error = first_error.clone();
                        async move {
                            match &result {
                                Ok(chunk) => {
                                    if let Some(ref usage) = chunk.usage {
                                        record_tokens(
                                            &ctx,
                                            usage.prompt_tokens,
                                            usage.completion_tokens,
                                        );
                                        tokens_in.store(usage.prompt_tokens, Ordering::Relaxed);
                                        tokens_out
                                            .store(usage.completion_tokens, Ordering::Relaxed);
                                        cache_hit.store(
                                            usage.prompt_cache_hit_tokens.unwrap_or(0),
                                            Ordering::Relaxed,
                                        );
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
                                tokens_in_done,
                                tokens_out_done,
                                cache_hit_done,
                                errored_done,
                                first_error_done,
                            )),
                        ),
                        |(mut inner, mut slot)| async move {
                            match inner.next().await {
                                Some(item) => Some((item, (inner, slot))),
                                None => {
                                    if let Some((state, ctx, ti, to, ch, er, fe)) = slot.take() {
                                        let errored = er.load(Ordering::Relaxed);
                                        record_duration(&ctx, if errored { "5xx" } else { "2xx" });
                                        let error = fe.lock().unwrap().take();
                                        let status = if errored { 0 } else { 200 };
                                        emit_usage(
                                            &state,
                                            &ctx,
                                            ENDPOINT,
                                            RequestOutcome {
                                                tokens_in: ti.load(Ordering::Relaxed),
                                                tokens_out: to.load(Ordering::Relaxed),
                                                cache_hit_tokens: ch.load(Ordering::Relaxed),
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

    let request = match deserialize_request(&raw_body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    for ext in state.extensions.iter() {
        if let Some(cached) = ext.on_cache_lookup(&request).await
            && let Ok(resp) = from_chat_completion(cached)
        {
            return Json(resp).into_response();
        }
    }

    let mut last_err = None;
    for deployment in &deployments {
        match try_chat_with_retries(deployment, &request).await {
            Ok(resp) => {
                let (pt, ct, ch) = resp
                    .usage
                    .as_ref()
                    .map(|u| {
                        (
                            u.prompt_tokens,
                            u.completion_tokens,
                            u.prompt_cache_hit_tokens.unwrap_or(0),
                        )
                    })
                    .unwrap_or((0, 0, 0));
                if pt > 0 || ct > 0 {
                    record_tokens(&ctx, pt, ct);
                }
                record_duration(&ctx, "2xx");
                emit_usage(&state, &ctx, ENDPOINT, RequestOutcome::ok(pt, ct, ch));
                for ext in state.extensions.iter() {
                    ext.on_response(&ctx, &request, &resp).await;
                }
                return match from_chat_completion(resp) {
                    Ok(anthropic) => Json(anthropic).into_response(),
                    Err(e) => {
                        for ext in state.extensions.iter() {
                            ext.on_error(&ctx, &e).await;
                        }
                        record_duration(&ctx, error_status(&e));
                        emit_usage_error(&state, &ctx, ENDPOINT, &e);
                        error_response(e)
                    }
                };
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
async fn handle_raw_anthropic<S: Storage, P: Provider>(
    state: &AppState<S, P>,
    principal: Principal,
    model: &str,
    deployments: &[&crabllm_provider::Deployment<P>],
    raw_body: axum::body::Bytes,
) -> Response {
    use crate::handlers::with_timeout;

    #[derive(serde::Deserialize)]
    struct AnthropicUsagePeek {
        usage: Option<AnthropicUsageFields>,
    }
    #[derive(serde::Deserialize)]
    struct AnthropicUsageFields {
        #[serde(default)]
        input_tokens: u32,
        #[serde(default)]
        output_tokens: u32,
        #[serde(default)]
        cache_read_input_tokens: Option<u32>,
    }

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

    let mut last_err = None;
    for deployment in deployments {
        match with_timeout(
            deployment.timeout,
            deployment.provider.anthropic_messages_raw(raw_body.clone()),
        )
        .await
        {
            Ok(resp_bytes) => {
                let (pt, ct, ch) =
                    crabllm_core::json::from_slice::<AnthropicUsagePeek>(&resp_bytes)
                        .ok()
                        .and_then(|p| p.usage)
                        .map(|u| {
                            (
                                u.input_tokens,
                                u.output_tokens,
                                u.cache_read_input_tokens.unwrap_or(0),
                            )
                        })
                        .unwrap_or((0, 0, 0));
                if pt > 0 || ct > 0 {
                    record_tokens(&ctx, pt, ct);
                }
                record_duration(&ctx, "2xx");
                emit_usage(state, &ctx, ENDPOINT, RequestOutcome::ok(pt, ct, ch));
                return (
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    resp_bytes,
                )
                    .into_response();
            }
            Err(e) => {
                if !e.is_transient() {
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
    let tokens_in = Arc::new(AtomicU32::new(0));
    let tokens_out = Arc::new(AtomicU32::new(0));
    let cache_hit = Arc::new(AtomicU32::new(0));
    let errored = Arc::new(AtomicBool::new(false));
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let ctx_c = ctx.clone();
    let tokens_in_c = tokens_in.clone();
    let tokens_out_c = tokens_out.clone();
    let cache_hit_c = cache_hit.clone();
    let errored_c = errored.clone();
    let first_error_c = first_error.clone();

    let events = anthropic_raw_sse(byte_stream);

    let sse_stream = events.map(move |result| match result {
        Ok((event_name, data)) => {
            peek_anthropic_usage(
                &event_name,
                &data,
                &tokens_in_c,
                &tokens_out_c,
                &cache_hit_c,
            );
            if tokens_in_c.load(Ordering::Relaxed) > 0 || tokens_out_c.load(Ordering::Relaxed) > 0 {
                record_tokens(
                    &ctx_c,
                    tokens_in_c.load(Ordering::Relaxed),
                    tokens_out_c.load(Ordering::Relaxed),
                );
            }
            Ok::<_, std::convert::Infallible>(Event::default().event(event_name).data(data))
        }
        Err(e) => {
            errored_c.store(true, Ordering::Relaxed);
            {
                let mut slot = first_error_c.lock().unwrap();
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
            Some((
                state,
                ctx,
                tokens_in,
                tokens_out,
                cache_hit,
                errored,
                first_error,
            )),
        ),
        |(mut inner, mut slot)| async move {
            match inner.next().await {
                Some(item) => Some((item, (inner, slot))),
                None => {
                    if let Some((state, ctx, ti, to, ch, er, fe)) = slot.take() {
                        let errored = er.load(Ordering::Relaxed);
                        record_duration(&ctx, if errored { "5xx" } else { "2xx" });
                        let error = fe.lock().unwrap().take();
                        let status = if errored { 0 } else { 200 };
                        emit_usage(
                            &state,
                            &ctx,
                            ENDPOINT,
                            RequestOutcome {
                                tokens_in: ti.load(Ordering::Relaxed),
                                tokens_out: to.load(Ordering::Relaxed),
                                cache_hit_tokens: ch.load(Ordering::Relaxed),
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

/// Extract usage tokens from Anthropic SSE event data.
fn peek_anthropic_usage(
    event_name: &str,
    data: &str,
    tokens_in: &AtomicU32,
    tokens_out: &AtomicU32,
    cache_hit: &AtomicU32,
) {
    let val: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return,
    };

    match event_name {
        "message_start" => {
            // Anthropic's `input_tokens` is only the uncached portion;
            // cache reads/creates are additive separate fields. Sum them so
            // `tokens_in` carries the OpenAI-style total prompt size, keeping
            // `cache_hit <= tokens_in` for downstream billing.
            if let Some(usage) = val.pointer("/message/usage") {
                let input = usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let cache_read = usage
                    .get("cache_read_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let cache_create = usage
                    .get("cache_creation_input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                tokens_in.store((input + cache_read + cache_create) as u32, Ordering::Relaxed);
                cache_hit.store(cache_read as u32, Ordering::Relaxed);
            }
        }
        "message_delta" => {
            if let Some(usage) = val.get("usage")
                && let Some(n) = usage.get("output_tokens").and_then(|v| v.as_u64())
            {
                tokens_out.store(n as u32, Ordering::Relaxed);
            }
        }
        _ => {}
    }
}

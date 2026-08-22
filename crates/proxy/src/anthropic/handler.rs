//! HTTP handler for `POST /v1/messages` (Anthropic-compatible).
//!
//! All requests are forwarded as raw Anthropic-format bytes to
//! Anthropic-compatible upstreams — no format translation. Non-compatible
//! deployments are skipped.

use crate::{
    AppState,
    auth::Principal,
    handlers::{
        RequestOutcome, emit_usage, emit_usage_error, error_response, error_status, first_provider,
        record_duration, record_tokens, try_anthropic_stream_with_retries, with_timeout,
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
    if !principal.can_use(&model) {
        return crate::handlers::model_forbidden(&model);
    }
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

    if is_stream {
        return handle_stream(&state, principal, &model, &deployments, raw_body).await;
    }
    handle_raw_anthropic(&state, principal, &model, &deployments, raw_body).await
}

/// Streaming raw byte proxy for Anthropic-compatible providers.
async fn handle_stream<S, P>(
    state: &AppState<S, P>,
    principal: Principal,
    model: &str,
    deployments: &[&crabllm_provider::Deployment<P>],
    raw_body: axum::body::Bytes,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let mut ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.to_string(),
        provider: first_provider(deployments),
        principal: principal.name,
        is_stream: true,
        started_at: Instant::now(),
    };

    for ext in state.extensions.iter() {
        if let Err(ext_err) = ext.on_request(&ctx).await {
            return crate::ext_response(ext_err);
        }
    }

    let mut last_err = None;
    for deployment in deployments {
        if !deployment.provider.is_anthropic_compat() {
            continue;
        }
        ctx.provider = deployment.name.clone();
        match try_anthropic_stream_with_retries(deployment, raw_body.clone()).await {
            Ok(byte_stream) => {
                return raw_anthropic_stream_response(byte_stream, state, ctx);
            }
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    }

    let e = last_err.unwrap_or_else(|| {
        crabllm_core::Error::Routing("no compatible providers available".into())
    });
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, "5xx");
    emit_usage_error(state, &ctx, ENDPOINT, &e);
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
    let mut ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.to_string(),
        provider: first_provider(deployments),
        principal: principal.name,
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
        if !deployment.provider.is_anthropic_compat() {
            continue;
        }
        ctx.provider = deployment.name.clone();
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

    let e = last_err.unwrap_or_else(|| {
        crabllm_core::Error::Routing("no compatible providers available".to_string())
    });
    for ext in state.extensions.iter() {
        ext.on_error(&ctx, &e).await;
    }
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, ENDPOINT, &e);
    error_response(e)
}

/// Streaming raw byte proxy response builder.
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
                record_tokens(
                    &ctx_c,
                    snapshot.prompt_tokens(),
                    snapshot.completion_tokens(),
                );
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
                            Err(crabllm_core::Error::Network(e.to_string())),
                            (bytes, buf, event_name, data),
                        ));
                    }
                    None => return None,
                }
            }
        },
    )
}

fn peek_anthropic_usage(event_name: &str, data: &str, usage: &Mutex<crabllm_core::Usage>) {
    let val: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return,
    };

    match event_name {
        "message_start" => {
            if let Some(u) = val.pointer("/message/usage") {
                let mut snap = usage.lock();
                snap.input_tokens =
                    u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
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

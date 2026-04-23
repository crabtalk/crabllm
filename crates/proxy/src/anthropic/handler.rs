//! HTTP handler for `POST /v1/messages` (Anthropic-compatible).
//!
//! Translates the inbound `AnthropicRequest` to the internal
//! `ChatCompletionRequest`, dispatches via the same provider pipeline the
//! OpenAI `/v1/chat/completions` handler uses, then translates the response
//! (or streaming chunk sequence) back to Anthropic's wire format.

use crate::{
    AppState,
    anthropic::{from_chat_completion, to_anthropic_sse, to_chat_completion},
    auth::KeyName,
    handlers::{
        emit_usage, emit_usage_error, error_response, error_status, record_duration, record_tokens,
        try_chat_with_retries, try_stream_with_retries,
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

/// POST /v1/messages
pub async fn messages<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(key_name): Extension<KeyName>,
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
        return handle_raw_anthropic(&state, key_name, &model, &deployments, raw_body).await;
    }

    // Full deserialization + translation for streaming, extensions, or
    // non-Anthropic upstreams.
    let anthropic_req: crabllm_core::AnthropicRequest = match crabllm_core::json::from_slice(&raw_body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(e.to_string(), "invalid_request_error")),
            )
                .into_response();
        }
    };
    let mut request = to_chat_completion(anthropic_req);

    let provider_name = registry
        .provider_name(&model)
        .unwrap_or_default()
        .to_string();

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: provider_name,
        key_name: key_name.0,
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
        request
            .extra
            .entry("stream_options".to_string())
            .or_insert(serde_json::json!({ "include_usage": true }));

        let mut last_err = None;
        for deployment in &deployments {
            match try_stream_with_retries(deployment, &request).await {
                Ok(stream) => {
                    let extensions = state.extensions.clone();
                    let ctx = Arc::new(ctx);
                    let errored = Arc::new(AtomicBool::new(false));
                    let tokens_in = Arc::new(AtomicU32::new(0));
                    let tokens_out = Arc::new(AtomicU32::new(0));
                    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

                    let ctx_done = ctx.clone();
                    let errored_done = errored.clone();
                    let tokens_in_done = tokens_in.clone();
                    let tokens_out_done = tokens_out.clone();
                    let first_error_done = first_error.clone();

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
                            // Anthropic documents `api_error` / `overloaded_error`
                            // rather than `server_error` in the enum.
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

                    // Wrap the stream so that emit_usage/record_duration run
                    // exactly once, when the wire stream drains — without
                    // putting a spurious SSE frame on the wire.
                    let finalized = futures::stream::unfold(
                        (
                            Box::pin(sse_stream),
                            Some((
                                state.clone(),
                                ctx_done,
                                tokens_in_done,
                                tokens_out_done,
                                errored_done,
                                first_error_done,
                            )),
                        ),
                        |(mut inner, mut slot)| async move {
                            match inner.next().await {
                                Some(item) => Some((item, (inner, slot))),
                                None => {
                                    if let Some((state, ctx, ti, to, er, fe)) = slot.take() {
                                        let errored = er.load(Ordering::Relaxed);
                                        record_duration(&ctx, if errored { "5xx" } else { "2xx" });
                                        let error = fe.lock().unwrap().take();
                                        let status = if errored { 0 } else { 200 };
                                        emit_usage(
                                            &state,
                                            &ctx,
                                            ENDPOINT,
                                            ti.load(Ordering::Relaxed),
                                            to.load(Ordering::Relaxed),
                                            status,
                                            error,
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

    // Non-streaming: cache first. A cache hit whose shape can't be translated
    // (missing usage, zero choices) falls through to live dispatch rather than
    // erroring — the cached payload was valid for the OpenAI handler, and the
    // translation gap is our bug, not the client's.
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
                let (pt, ct) = resp
                    .usage
                    .as_ref()
                    .map(|u| (u.prompt_tokens, u.completion_tokens))
                    .unwrap_or((0, 0));
                if pt > 0 || ct > 0 {
                    record_tokens(&ctx, pt, ct);
                }
                record_duration(&ctx, "2xx");
                emit_usage(&state, &ctx, ENDPOINT, pt, ct, 200, None);
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
    key_name: KeyName,
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
        key_name: key_name.0,
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
                let (pt, ct) = crabllm_core::json::from_slice::<AnthropicUsagePeek>(&resp_bytes)
                    .ok()
                    .and_then(|p| p.usage)
                    .map(|u| (u.input_tokens, u.output_tokens))
                    .unwrap_or((0, 0));
                if pt > 0 || ct > 0 {
                    record_tokens(&ctx, pt, ct);
                }
                record_duration(&ctx, "2xx");
                emit_usage(state, &ctx, ENDPOINT, pt, ct, 200, None);
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

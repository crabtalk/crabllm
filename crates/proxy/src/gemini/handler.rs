//! HTTP handler for `POST /v1beta/models/{model}:generateContent`.
//!
//! Both action variants (`:generateContent`, `:streamGenerateContent`) route
//! through this handler. The path segment is split on `:` to extract model
//! and action. The query string is consulted only for `?alt=sse` (Google's
//! streaming format selector).
//!
//! Limitation: translated streaming (non-gemini-compat upstream) is not yet
//! implemented — the `ChatCompletionChunk` → Gemini SSE shape conversion is
//! a follow-up. Non-streaming translates fine through the trait default impl.

use crate::{
    AppState,
    auth::Principal,
    handlers::{
        RequestOutcome, emit_usage, emit_usage_error, error_response, error_status,
        record_duration, record_tokens, with_timeout,
    },
};
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use bytes::Bytes;
use crabllm_core::{ApiError, GeminiRequest, Provider, RequestContext, Storage};
use futures::StreamExt;
use parking_lot::Mutex;
use std::{sync::Arc, time::Instant};

const ENDPOINT: &str = "gemini.generateContent";

/// POST /v1beta/models/{model_action}
pub async fn generate_content<S, P>(
    State(state): State<AppState<S, P>>,
    Extension(principal): Extension<Principal>,
    Path(model_action): Path<String>,
    raw_body: Bytes,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let Some((model_raw, action)) = model_action.split_once(':') else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!("invalid path '{model_action}', expected '<model>:<action>'"),
                "invalid_request_error",
            )),
        )
            .into_response();
    };
    let is_stream = match action {
        "generateContent" => false,
        "streamGenerateContent" => true,
        other => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("unknown Gemini action '{other}'"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let registry = state.registry();
    let model = registry.resolve(model_raw).to_string();
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
        is_stream,
        started_at: Instant::now(),
    };

    if is_stream {
        return stream_path(&state, ctx, &model, &deployments, raw_body).await;
    }
    unary_path(&state, ctx, &model, &deployments, raw_body).await
}

async fn unary_path<S, P>(
    state: &AppState<S, P>,
    ctx: RequestContext,
    model: &str,
    deployments: &[&crabllm_provider::Deployment<P>],
    raw_body: Bytes,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let mut last_err = None;
    for deployment in deployments {
        if deployment.provider.is_gemini_compat() {
            match with_timeout(
                deployment.timeout,
                deployment
                    .provider
                    .gemini_generate_content_raw(model, raw_body.clone()),
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
                    continue;
                }
            }
        }

        // Translated path: deserialize → typed call → serialize.
        let request: GeminiRequest = match crabllm_core::json::from_slice(&raw_body) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(e.to_string(), "invalid_request_error")),
                )
                    .into_response();
            }
        };
        match with_timeout(
            deployment.timeout,
            deployment.provider.gemini_generate_content(model, &request),
        )
        .await
        {
            Ok(resp) => {
                let usage = resp
                    .usage_metadata
                    .as_ref()
                    .map(crabllm_core::Usage::from)
                    .unwrap_or_default();
                if usage.prompt_tokens() > 0 || usage.completion_tokens() > 0 {
                    record_tokens(&ctx, usage.prompt_tokens(), usage.completion_tokens());
                }
                record_duration(&ctx, "2xx");
                emit_usage(state, &ctx, ENDPOINT, RequestOutcome::ok(usage));
                return Json(resp).into_response();
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

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, ENDPOINT, &e);
    error_response(e)
}

async fn stream_path<S, P>(
    state: &AppState<S, P>,
    ctx: RequestContext,
    model: &str,
    deployments: &[&crabllm_provider::Deployment<P>],
    raw_body: Bytes,
) -> Response
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let mut last_err = None;
    for deployment in deployments {
        if deployment.provider.is_gemini_compat() {
            match with_timeout(
                deployment.timeout,
                deployment
                    .provider
                    .gemini_generate_content_stream_raw(model, raw_body.clone()),
            )
            .await
            {
                Ok(byte_stream) => {
                    record_duration(&ctx, "2xx");
                    // Upstream already sends SSE-shape bytes (`data: <json>\n\n`).
                    // Forward verbatim; metrics for translated chunks would
                    // require parsing the SSE — defer to a follow-up.
                    let body = axum::body::Body::from_stream(byte_stream);
                    return axum::http::Response::builder()
                        .status(StatusCode::OK)
                        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                        .header("cache-control", "no-cache")
                        .body(body)
                        .unwrap();
                }
                Err(e) => {
                    if !e.is_transient() {
                        record_duration(&ctx, error_status(&e));
                        emit_usage_error(state, &ctx, ENDPOINT, &e);
                        return error_response(e);
                    }
                    last_err = Some(e);
                    continue;
                }
            }
        }

        // Translated streaming for non-compat providers.
        let request: GeminiRequest = match crabllm_core::json::from_slice(&raw_body) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(e.to_string(), "invalid_request_error")),
                )
                    .into_response();
            }
        };
        match with_timeout(
            deployment.timeout,
            deployment
                .provider
                .gemini_generate_content_stream(model, &request),
        )
        .await
        {
            Ok(stream) => {
                let ctx = Arc::new(ctx);
                let usage: Arc<Mutex<crabllm_core::Usage>> =
                    Arc::new(Mutex::new(crabllm_core::Usage::default()));

                let usage_c = usage.clone();
                let ctx_c = ctx.clone();

                let sse_stream = stream.map(move |result| match result {
                    Ok(resp) => {
                        if let Some(u) = resp.usage_metadata.as_ref() {
                            let canonical = crabllm_core::Usage::from(u);
                            record_tokens(
                                &ctx_c,
                                canonical.prompt_tokens(),
                                canonical.completion_tokens(),
                            );
                            *usage_c.lock() = canonical;
                        }
                        let json = crabllm_core::json::to_string(&resp).unwrap_or_default();
                        Ok::<_, std::convert::Infallible>(Event::default().data(json))
                    }
                    Err(e) => {
                        let json = crabllm_core::json::to_string(&serde_json::json!({
                            "error": { "message": e.to_string() }
                        }))
                        .unwrap_or_default();
                        Ok(Event::default().data(json))
                    }
                });

                let state_clone = state.clone();
                let ctx_done = ctx.clone();
                let usage_done = usage.clone();

                let finalized = futures::stream::unfold(
                    (Box::pin(sse_stream), Some((state_clone, ctx_done, usage_done))),
                    |(mut inner, mut slot)| async move {
                        match inner.next().await {
                            Some(item) => Some((item, (inner, slot))),
                            None => {
                                if let Some((st, cx, u)) = slot.take() {
                                    let usage = u.lock().clone();
                                    record_duration(&cx, "2xx");
                                    emit_usage(&st, &cx, ENDPOINT, RequestOutcome::ok(usage));
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
            Err(e) => {
                if !e.is_transient() {
                    record_duration(&ctx, error_status(&e));
                    emit_usage_error(state, &ctx, ENDPOINT, &e);
                    return error_response(e);
                }
                last_err = Some(e);
                continue;
            }
        }
    }

    let e =
        last_err.unwrap_or_else(|| crabllm_core::Error::Internal("no providers available".into()));
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, ENDPOINT, &e);
    error_response(e)
}


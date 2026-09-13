//! HTTP handler for `POST /v1beta/models/{model}:generateContent`.
//!
//! Both action variants (`:generateContent`, `:streamGenerateContent`) route
//! through this handler. All requests are forwarded as raw Gemini-format bytes
//! to Gemini-compatible upstreams — no format translation. Non-compatible
//! deployments are skipped.

use crate::{
    AppState,
    auth::Principal,
    handlers::{
        RequestOutcome, emit_usage, emit_usage_error, error_response, error_status, first_provider,
        record_duration, record_tokens, with_timeout,
    },
};
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use crabllm_core::{ApiError, Provider, RequestContext, Storage};
use std::time::Instant;

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

    let ctx = RequestContext {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: model.clone(),
        provider: first_provider(&deployments),
        principal: principal.name,
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
    mut ctx: RequestContext,
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
        if !deployment.provider.is_gemini_compat() {
            continue;
        }
        ctx.provider = deployment.name.clone();
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
            }
        }
    }

    let e = last_err.unwrap_or_else(|| {
        crabllm_core::Error::Routing("no compatible providers available".into())
    });
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, ENDPOINT, &e);
    error_response(e)
}

async fn stream_path<S, P>(
    state: &AppState<S, P>,
    mut ctx: RequestContext,
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
        if !deployment.provider.is_gemini_compat() {
            continue;
        }
        ctx.provider = deployment.name.clone();
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

    let e = last_err.unwrap_or_else(|| {
        crabllm_core::Error::Routing("no compatible providers available".into())
    });
    record_duration(&ctx, error_status(&e));
    emit_usage_error(state, &ctx, ENDPOINT, &e);
    error_response(e)
}

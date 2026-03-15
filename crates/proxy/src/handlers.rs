use crate::{AppState, auth::KeyName};
use axum::{
    Extension, Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use crabtalk_core::{
    ApiError, ChatCompletionRequest, EmbeddingRequest, Model, ModelList, RequestContext, Storage,
};
use futures::StreamExt;
use std::time::Instant;

/// POST /v1/chat/completions
pub async fn chat_completions<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let provider = match state.registry.get(&request.model) {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{}' not found", request.model),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&request.model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: String::new(),
        model: request.model.clone(),
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

    // Branch on stream field.
    if ctx.is_stream {
        match provider
            .chat_completion_stream(&state.client, &request)
            .await
        {
            Ok(stream) => {
                let extensions = state.extensions.clone();
                let ctx_clone = ctx.clone();

                // Wrap stream with extension chunk observation.
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
                        let json =
                            serde_json::to_string(&ApiError::new(e.to_string(), "server_error"))
                                .unwrap_or_default();
                        Ok(Event::default().data(json))
                    }
                });

                // Append [DONE] sentinel after the stream ends.
                let done = futures::stream::once(async {
                    Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                });
                let full_stream = sse_stream.chain(done);

                Sse::new(full_stream)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response()
            }
            Err(e) => {
                for ext in state.extensions.iter() {
                    ext.on_error(&ctx, &e).await;
                }
                error_response(e)
            }
        }
    } else {
        match provider.chat_completion(&state.client, &request).await {
            Ok(resp) => {
                for ext in state.extensions.iter() {
                    ext.on_response(&ctx, &resp).await;
                }
                Json(resp).into_response()
            }
            Err(e) => {
                for ext in state.extensions.iter() {
                    ext.on_error(&ctx, &e).await;
                }
                error_response(e)
            }
        }
    }
}

/// POST /v1/embeddings
pub async fn embeddings<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    Extension(key_name): Extension<KeyName>,
    Json(request): Json<EmbeddingRequest>,
) -> Response {
    let provider = match state.registry.get(&request.model) {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{}' not found", request.model),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let provider_name = state
        .config
        .models()
        .get(&request.model)
        .cloned()
        .unwrap_or_default();

    let ctx = RequestContext {
        request_id: String::new(),
        model: request.model.clone(),
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

    match provider.embedding(&state.client, &request).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => {
            for ext in state.extensions.iter() {
                ext.on_error(&ctx, &e).await;
            }
            error_response(e)
        }
    }
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

/// Map a provider Error to an HTTP error response.
fn error_response(e: crabtalk_core::Error) -> Response {
    let (status, api_error) = match &e {
        crabtalk_core::Error::Provider { status, body } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            ApiError::new(body.clone(), "upstream_error"),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::new(e.to_string(), "server_error"),
        ),
    };
    (status, Json(api_error)).into_response()
}

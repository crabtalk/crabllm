use crate::AppState;
use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use crabllm_core::{ApiError, Provider, Storage};

/// Wrapper for the authenticated key name, inserted into request extensions.
#[derive(Clone, Debug)]
pub struct KeyName(pub Option<String>);

/// Auth middleware: validates Bearer token against configured virtual keys.
/// Skips auth only when no admin_token is configured AND key_map is empty.
/// Inserts `KeyName` into request extensions for downstream handlers.
pub async fn auth<S: Storage + 'static, P: Provider + 'static>(
    State(state): State<AppState<S, P>>,
    mut request: Request,
    next: Next,
) -> Response {
    // Skip auth when key management is disabled and no keys exist.
    if state.config.admin_token.is_none()
        && state
            .key_map
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    {
        request.extensions_mut().insert(KeyName(None));
        return next.run(request).await;
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header.and_then(|h| h.strip_prefix("Bearer ")) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ApiError::new(
                    "missing or invalid Authorization header",
                    "authentication_error",
                )),
            )
                .into_response();
        }
    };

    let key_name = state
        .key_map
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(token)
        .cloned();

    let Some(key_name) = key_name else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ApiError::new("invalid API key", "authentication_error")),
        )
            .into_response();
    };

    request.extensions_mut().insert(KeyName(Some(key_name)));

    next.run(request).await
}

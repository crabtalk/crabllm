use crate::AppState;
use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use crabtalk_core::{ApiError, Storage};

/// Wrapper for the authenticated key name, inserted into request extensions.
#[derive(Clone, Debug)]
pub struct KeyName(pub Option<String>);

/// Auth middleware: validates Bearer token against configured virtual keys.
/// If no keys are configured, all requests pass through.
/// Inserts `KeyName` into request extensions for downstream handlers.
pub async fn auth<S: Storage + 'static>(
    State(state): State<AppState<S>>,
    mut request: Request,
    next: Next,
) -> Response {
    // If no keys configured, skip auth entirely.
    if state.config.keys.is_empty() {
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

    let Some(key_name) = state.key_map.get(token) else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ApiError::new("invalid API key", "authentication_error")),
        )
            .into_response();
    };

    request
        .extensions_mut()
        .insert(KeyName(Some(key_name.clone())));

    next.run(request).await
}

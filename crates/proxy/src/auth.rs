use crate::AppState;
use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use crabllm_core::{ApiError, Provider, Storage};

/// Opaque identity token attached by the authentication layer, inserted into
/// request extensions. Standalone deployments populate this from the configured
/// key name; embedders providing their own auth populate it with whatever
/// caller identifier they need to attribute work against. Treat as opaque —
/// do not parse, sanitize, or display without intentional formatting.
#[derive(Clone, Debug)]
pub struct Principal(pub Option<String>);

/// Auth middleware: validates Bearer token against configured virtual keys.
/// Skips auth only when no admin_token is configured AND key_map is empty.
/// Inserts `Principal` into request extensions for downstream handlers.
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
        request.extensions_mut().insert(Principal(None));
        return next.run(request).await;
    }

    // Accept either OpenAI-style `Authorization: Bearer <key>` or Anthropic-style
    // `x-api-key: <key>`. Both map to the same virtual-key lookup.
    let headers = request.headers();
    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));
    let x_api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());

    let token = match bearer.or(x_api_key) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ApiError::new(
                    "missing Authorization or x-api-key header",
                    "authentication_error",
                )),
            )
                .into_response();
        }
    };

    let principal = state
        .key_map
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(token)
        .cloned();

    let Some(principal) = principal else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ApiError::new("invalid API key", "authentication_error")),
        )
            .into_response();
    };

    request.extensions_mut().insert(Principal(Some(principal)));

    next.run(request).await
}

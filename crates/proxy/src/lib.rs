use axum::{
    Json, Router,
    extract::Request,
    middleware,
    response::Response,
    routing::{get, post},
};
use crabllm_core::{Prefix, Provider, Storage};

pub use auth::Principal;
pub use state::{AppState, UsageEvent};

// Storage table prefixes. Each 4-byte prefix namespaces a logical table
// in the key-value storage backend.
pub const PREFIX_KEYS: Prefix = *b"keys";
pub const PREFIX_RATE_LIMIT: Prefix = *b"rlim";
pub const PREFIX_USAGE: Prefix = *b"usge";
pub const PREFIX_CACHE: Prefix = *b"cach";
pub const PREFIX_BUDGET: Prefix = *b"bdgt";
pub const PREFIX_AUDIT: Prefix = *b"alog";
pub const PREFIX_PROVIDERS: Prefix = *b"prvd";

pub mod admin;
pub mod admin_providers;
pub mod anthropic;
pub mod auth;
pub mod ext;
pub mod handlers;
#[cfg(feature = "openapi")]
pub mod openapi;
mod state;
pub mod storage;

/// Middleware that tracks the number of in-flight API requests.
/// For SSE streams, the gauge decrements when the response starts (not when the
/// stream ends), so it undercounts long-lived streaming connections.
async fn track_active_connections(request: Request, next: middleware::Next) -> Response {
    metrics::gauge!("crabllm_active_connections").increment(1.0);
    let response = next.run(request).await;
    metrics::gauge!("crabllm_active_connections").decrement(1.0);
    response
}

/// Middleware that logs every incoming HTTP request with method, path,
/// status, and latency. 2xx/3xx log at `info`, 4xx/5xx at `warn`.
/// `/health` and `/metrics` log at `debug` so probes don't flood the
/// default output.
pub async fn log_request(request: Request, next: middleware::Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = std::time::Instant::now();

    let response = next.run(request).await;
    let status = response.status();
    let latency_ms = start.elapsed().as_millis() as u64;
    let is_probe = path == "/health" || path == "/metrics";

    if is_probe {
        tracing::debug!(%method, path, status = status.as_u16(), latency_ms, "request");
    } else if status.is_client_error() || status.is_server_error() {
        tracing::warn!(%method, path, status = status.as_u16(), latency_ms, "request");
    } else {
        tracing::info!(%method, path, status = status.as_u16(), latency_ms, "request");
    }

    response
}

/// Build the bare API route tree (`/v1/*`) bound to `state`, with **no
/// auth, observability, or admin layers**. Embedders use this when they
/// want to install their own auth or observability middleware instead of
/// the defaults in [`router`]. Standalone callers should prefer [`router`].
pub fn routes<S, P>(state: AppState<S, P>) -> Router
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    Router::<AppState<S, P>>::new()
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions::<S, P>),
        )
        .route("/v1/messages", post(anthropic::messages::<S, P>))
        .route("/v1/embeddings", post(handlers::embeddings::<S, P>))
        .route(
            "/v1/images/generations",
            post(handlers::image_generations::<S, P>),
        )
        .route("/v1/audio/speech", post(handlers::audio_speech::<S, P>))
        .route(
            "/v1/audio/transcriptions",
            post(handlers::audio_transcriptions::<S, P>),
        )
        .route("/v1/models", get(handlers::models::<S, P>))
        .route("/v1/usage", get(handlers::usage::<S, P>))
        .with_state(state)
}

/// Build the Axum router with all API routes, the built-in auth middleware,
/// active-connection tracking, `/health`, and merged admin routes. This is
/// the batteries-included shape used by the standalone gateway binary.
pub fn router<S, P>(state: AppState<S, P>, admin_routes: Vec<Router>) -> Router
where
    S: Storage + 'static,
    P: Provider + 'static,
{
    let mut app = routes(state.clone())
        .layer(middleware::from_fn_with_state(
            state,
            auth::auth::<S, P>,
        ))
        .layer(middleware::from_fn(track_active_connections));

    // Health check — outside auth middleware so load balancers can probe it.
    app = app.route(
        "/health",
        get(|| async { Json(serde_json::json!({"status": "ok"})) }),
    );

    // Merge extension-provided admin routes (stateless — extensions
    // capture their own state via closures in the Router<()>).
    for admin_router in admin_routes {
        app = app.merge(admin_router);
    }

    app
}

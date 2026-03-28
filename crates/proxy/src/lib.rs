use axum::{
    Json, Router,
    extract::Request,
    middleware,
    response::Response,
    routing::{get, post},
};
use crabllm_core::Storage;

pub use auth::KeyName;
pub use state::AppState;

pub mod auth;
pub mod ext;
mod handlers;
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

/// Build the Axum router with all API routes and admin routes.
pub fn router<S: Storage + 'static>(state: AppState<S>, admin_routes: Vec<Router>) -> Router {
    let mut app = Router::<AppState<S>>::new()
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions::<S>),
        )
        .route("/v1/embeddings", post(handlers::embeddings::<S>))
        .route(
            "/v1/images/generations",
            post(handlers::image_generations::<S>),
        )
        .route("/v1/audio/speech", post(handlers::audio_speech::<S>))
        .route(
            "/v1/audio/transcriptions",
            post(handlers::audio_transcriptions::<S>),
        )
        .route("/v1/models", get(handlers::models::<S>))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth::<S>,
        ))
        .layer(middleware::from_fn(track_active_connections))
        .with_state(state);

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

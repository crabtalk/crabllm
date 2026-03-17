use axum::{
    Router, middleware,
    routing::{get, post},
};
use crabtalk_core::Storage;

pub use auth::KeyName;
pub use state::AppState;

pub mod auth;
pub mod ext;
mod handlers;
mod state;
pub mod storage;

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
        .with_state(state);

    // Merge extension-provided admin routes (stateless — extensions
    // capture their own state via closures in the Router<()>).
    for admin_router in admin_routes {
        app = app.merge(admin_router);
    }

    app
}

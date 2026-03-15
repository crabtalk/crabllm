use axum::{
    Router, middleware,
    routing::{get, post},
};

pub use auth::KeyName;
pub use state::AppState;

pub mod auth;
pub mod ext;
mod handlers;
mod state;

/// Build the Axum router with all API routes and extension routes.
pub fn router(state: AppState) -> Router {
    let mut app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/models", get(handlers::models))
        .layer(middleware::from_fn_with_state(state.clone(), auth::auth))
        .with_state(state.clone());

    // Merge extension-provided admin routes (stateless — extensions
    // capture their own state via closures in the Router<()>).
    for ext in state.extensions.iter() {
        if let Some(ext_router) = ext.routes() {
            app = app.merge(ext_router);
        }
    }

    app
}

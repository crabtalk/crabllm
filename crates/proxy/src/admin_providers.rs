use arc_swap::ArcSwap;
use axum::{
    Json, Router,
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
};
use crabllm_core::{ApiError, Error, GatewayConfig, Provider};
use crabllm_provider::ProviderRegistry;
use serde::Serialize;
use std::{path::PathBuf, sync::Arc};

/// A closure that rebuilds the provider registry from a config.
/// The binary provides this because the proxy crate doesn't know the
/// concrete `P` construction path (e.g. `Dispatch::Remote`).
pub type Rebuilder<P> =
    Arc<dyn Fn(&GatewayConfig) -> Result<ProviderRegistry<P>, Error> + Send + Sync>;

struct ProviderAdminState<P: Provider> {
    registry: Arc<ArcSwap<ProviderRegistry<P>>>,
    config_path: PathBuf,
    admin_token: String,
    rebuilder: Rebuilder<P>,
}

impl<P: Provider> Clone for ProviderAdminState<P> {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            config_path: self.config_path.clone(),
            admin_token: self.admin_token.clone(),
            rebuilder: self.rebuilder.clone(),
        }
    }
}

/// Build admin provider management routes, protected by admin token auth.
pub fn provider_admin_routes<P: Provider + 'static>(
    registry: Arc<ArcSwap<ProviderRegistry<P>>>,
    config_path: PathBuf,
    admin_token: String,
    rebuilder: Rebuilder<P>,
) -> Router {
    let state = ProviderAdminState {
        registry,
        config_path,
        admin_token,
        rebuilder,
    };
    Router::new()
        .route("/v1/admin/providers/reload", post(reload_providers::<P>))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            admin_auth::<P>,
        ))
        .with_state(state)
}

async fn admin_auth<P: Provider>(
    State(state): State<ProviderAdminState<P>>,
    request: Request,
    next: Next,
) -> Response {
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    match token {
        Some(t) if crate::admin::constant_time_eq(t, &state.admin_token) => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(ApiError::new(
                "missing or invalid admin token",
                "authentication_error",
            )),
        )
            .into_response(),
    }
}

#[derive(Serialize)]
struct ReloadResponse {
    status: &'static str,
    models: usize,
    providers: usize,
}

/// POST /v1/admin/providers/reload — re-read config from disk and
/// atomically swap the provider registry.
async fn reload_providers<P: Provider>(State(state): State<ProviderAdminState<P>>) -> Response {
    let raw = match std::fs::read_to_string(&state.config_path) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new(
                    format!("failed to read config file: {e}"),
                    "server_error",
                )),
            )
                .into_response();
        }
    };

    let config: GatewayConfig = match toml::from_str(&raw) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("failed to parse config: {e}"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let new_registry = match (state.rebuilder)(&config) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("failed to build registry: {e}"),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    let models = new_registry.model_names().count();
    let providers = new_registry.provider_count();
    state.registry.store(Arc::new(new_registry));
    eprintln!("provider registry reloaded: {models} models, {providers} providers");

    Json(ReloadResponse {
        status: "ok",
        models,
        providers,
    })
    .into_response()
}

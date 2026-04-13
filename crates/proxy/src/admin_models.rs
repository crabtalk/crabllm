use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use crabllm_core::{
    ApiError, GatewayConfig, ModelInfo, Prefix, Storage, resolve_model_info_full, storage_key,
};
use serde::Serialize;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
};

const MODEL_PREFIX: Prefix = *b"modl";

#[derive(Clone)]
struct ModelAdminState {
    storage: Arc<dyn Storage>,
    model_overrides: Arc<RwLock<HashMap<String, ModelInfo>>>,
    config: GatewayConfig,
    config_path: PathBuf,
    admin_token: String,
}

/// Build admin model metadata routes, protected by admin token auth.
pub fn model_admin_routes(
    storage: Arc<dyn Storage>,
    model_overrides: Arc<RwLock<HashMap<String, ModelInfo>>>,
    config: GatewayConfig,
    config_path: PathBuf,
    admin_token: String,
) -> Router {
    let state = ModelAdminState {
        storage,
        model_overrides,
        config,
        config_path,
        admin_token,
    };
    Router::new()
        .route("/v1/admin/models", get(list_models))
        .route("/v1/admin/models/flush", post(flush_models))
        .route(
            "/v1/admin/models/{model}",
            put(upsert_model).get(get_model).delete(delete_model),
        )
        .route_layer(middleware::from_fn_with_state(state.clone(), admin_auth))
        .with_state(state)
}

async fn admin_auth(
    State(state): State<ModelAdminState>,
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
        _ => err_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid admin token",
            "authentication_error",
        ),
    }
}

fn err_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (status, Json(ApiError::new(message, error_type))).into_response()
}

#[derive(Serialize)]
struct ModelMetadataEntry {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pricing: Option<crabllm_core::PricingConfig>,
    source: &'static str,
}

/// GET /v1/admin/models — list all model metadata (admin + config + default).
async fn list_models(State(state): State<ModelAdminState>) -> Response {
    let admin_map = state
        .model_overrides
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    // Collect all known model names from all three layers.
    let mut all_models: HashMap<&str, &'static str> = HashMap::new();

    // Static defaults — iterate the config models and admin models to find
    // which ones have defaults, but also include any model registered in
    // the config or admin layer.
    for name in admin_map.keys() {
        all_models.insert(name, "admin");
    }
    for name in state.config.models.keys() {
        all_models.entry(name).or_insert("config");
    }

    // We can't enumerate all static defaults (no exported list), but any
    // model that appears in config or admin is already covered. Models that
    // only exist in the static table won't appear here — that's by design,
    // since they're not "registered" in this gateway instance.

    let mut entries: Vec<ModelMetadataEntry> = Vec::new();
    for (model, source) in &all_models {
        if let Some(info) = resolve_model_info_full(model, &admin_map, &state.config.models) {
            entries.push(ModelMetadataEntry {
                model: model.to_string(),
                context_length: info.context_length,
                pricing: info.pricing,
                source,
            });
        }
    }

    entries.sort_by(|a, b| a.model.cmp(&b.model));
    Json(entries).into_response()
}

/// GET /v1/admin/models/:model — resolved metadata for one model.
async fn get_model(State(state): State<ModelAdminState>, Path(model): Path<String>) -> Response {
    let admin_map = state
        .model_overrides
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    let source = if admin_map.contains_key(&model) {
        "admin"
    } else if state.config.models.contains_key(&model) {
        "config"
    } else {
        "default"
    };

    let Some(info) = resolve_model_info_full(&model, &admin_map, &state.config.models) else {
        return err_response(
            StatusCode::NOT_FOUND,
            &format!("model '{model}' not found"),
            "invalid_request_error",
        );
    };

    Json(ModelMetadataEntry {
        model,
        context_length: info.context_length,
        pricing: info.pricing,
        source,
    })
    .into_response()
}

/// PUT /v1/admin/models/:model — upsert a model metadata override.
async fn upsert_model(
    State(state): State<ModelAdminState>,
    Path(model): Path<String>,
    Json(info): Json<ModelInfo>,
) -> Response {
    // Persist to storage first.
    let skey = storage_key(&MODEL_PREFIX, model.as_bytes());
    let value = match serde_json::to_vec(&info) {
        Ok(v) => v,
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
    };
    if let Err(e) = state.storage.set(&skey, value).await {
        return err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        );
    }

    // Update in-memory map.
    {
        let mut map = state
            .model_overrides
            .write()
            .unwrap_or_else(|e| e.into_inner());
        map.insert(model.clone(), info);
    }

    // Return resolved metadata (merged with config + defaults).
    let admin_map = state
        .model_overrides
        .read()
        .unwrap_or_else(|e| e.into_inner());
    let resolved =
        resolve_model_info_full(&model, &admin_map, &state.config.models).unwrap_or_default();

    (
        StatusCode::OK,
        Json(ModelMetadataEntry {
            model,
            context_length: resolved.context_length,
            pricing: resolved.pricing,
            source: "admin",
        }),
    )
        .into_response()
}

/// DELETE /v1/admin/models/:model — remove an admin override.
async fn delete_model(State(state): State<ModelAdminState>, Path(model): Path<String>) -> Response {
    let exists = state
        .model_overrides
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&model);

    if !exists {
        return err_response(
            StatusCode::NOT_FOUND,
            &format!("no admin override for model '{model}'"),
            "invalid_request_error",
        );
    }

    // Storage-first: delete from storage before updating in-memory map.
    let skey = storage_key(&MODEL_PREFIX, model.as_bytes());
    if let Err(e) = state.storage.delete(&skey).await {
        return err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        );
    }

    state
        .model_overrides
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&model);

    StatusCode::NO_CONTENT.into_response()
}

/// POST /v1/admin/models/flush — write admin overrides into the TOML config file.
///
/// Merges admin overrides into the `[models]` section of the config file.
/// After flush, the overrides become config entries and survive independently
/// of the storage backend.
async fn flush_models(State(state): State<ModelAdminState>) -> Response {
    let admin_map = state
        .model_overrides
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    if admin_map.is_empty() {
        return err_response(
            StatusCode::BAD_REQUEST,
            "no admin overrides to flush",
            "invalid_request_error",
        );
    }

    // Read the current config from disk (not the stale startup clone)
    // to avoid overwriting manual edits made since startup.
    let mut config: GatewayConfig = match std::fs::read_to_string(&state.config_path) {
        Ok(raw) => match toml::from_str(&raw) {
            Ok(c) => c,
            Err(e) => {
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("failed to parse config file: {e}"),
                    "server_error",
                );
            }
        },
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to read config file: {e}"),
                "server_error",
            );
        }
    };

    // Merge admin overrides into the current config.
    for (model, info) in &admin_map {
        config.models.insert(model.clone(), info.clone());
    }

    // Serialize and write.
    let toml_str = match toml::to_string_pretty(&config) {
        Ok(s) => s,
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to serialize config: {e}"),
                "server_error",
            );
        }
    };

    if let Err(e) = std::fs::write(&state.config_path, &toml_str) {
        return err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to write config file: {e}"),
            "server_error",
        );
    }

    // Clear admin overrides from memory and storage — they're now in config.
    let flushed = admin_map.len();
    state
        .model_overrides
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .clear();

    for model in admin_map.keys() {
        let skey = storage_key(&MODEL_PREFIX, model.as_bytes());
        let _ = state.storage.delete(&skey).await;
    }

    #[derive(Serialize)]
    struct FlushResult {
        flushed: usize,
        path: String,
    }

    Json(FlushResult {
        flushed,
        path: state.config_path.display().to_string(),
    })
    .into_response()
}

/// Load model overrides from storage into the in-memory map.
pub async fn load_stored_models(
    storage: &dyn Storage,
    overrides: &RwLock<HashMap<String, ModelInfo>>,
) {
    let pairs = match storage.list(&MODEL_PREFIX).await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: failed to load stored model overrides: {e}");
            return;
        }
    };

    let mut map = overrides.write().unwrap_or_else(|e| e.into_inner());
    for (raw_key, value) in pairs {
        let suffix = &raw_key[crabllm_core::PREFIX_LEN..];
        let model = match std::str::from_utf8(suffix) {
            Ok(s) => s.to_string(),
            Err(_) => continue,
        };
        if let Ok(info) = serde_json::from_slice::<ModelInfo>(&value) {
            map.insert(model, info);
        }
    }
}

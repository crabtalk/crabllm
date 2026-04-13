use crate::PREFIX_KEYS;
use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use crabllm_core::{ApiError, KeyConfig, KeyRateLimit, Storage, storage_key};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};

#[derive(Clone)]
pub struct KeyAdminState {
    storage: Arc<dyn Storage>,
    key_map: Arc<RwLock<HashMap<String, String>>>,
    admin_token: String,
    toml_key_names: HashSet<String>,
    toml_keys: Vec<KeyConfig>,
}

/// Build admin key management routes, protected by admin token auth.
pub fn key_admin_routes(
    storage: Arc<dyn Storage>,
    key_map: Arc<RwLock<HashMap<String, String>>>,
    admin_token: String,
    toml_keys: Vec<KeyConfig>,
) -> Router {
    let toml_key_names: HashSet<String> = toml_keys.iter().map(|k| k.name.clone()).collect();
    let state = KeyAdminState {
        storage,
        key_map,
        admin_token,
        toml_key_names,
        toml_keys,
    };
    Router::new()
        .route("/v1/admin/keys", post(create_key).get(list_keys))
        .route(
            "/v1/admin/keys/{name}",
            get(get_key).patch(update_key).delete(delete_key),
        )
        .route_layer(middleware::from_fn_with_state(state.clone(), admin_auth))
        .with_state(state)
}

/// Timing-resistant token comparison. Leaks length but not content.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Validate admin Bearer token from request headers.
/// Returns `Ok(())` on success, `Err(Response)` with 401 on failure.
#[allow(clippy::result_large_err)]
pub(crate) fn check_admin_token(request: &Request, admin_token: &str) -> Result<(), Response> {
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    match token {
        Some(t) if constant_time_eq(t, admin_token) => Ok(()),
        _ => Err(err_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid admin token",
            "authentication_error",
        )),
    }
}

async fn admin_auth(State(state): State<KeyAdminState>, request: Request, next: Next) -> Response {
    if let Err(r) = check_admin_token(&request, &state.admin_token) {
        return r;
    }
    next.run(request).await
}

pub(crate) fn err_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (status, Json(ApiError::new(message, error_type))).into_response()
}

#[derive(Deserialize)]
struct CreateKeyRequest {
    name: String,
    #[serde(default = "default_models")]
    models: Vec<String>,
    #[serde(default)]
    rate_limit: Option<KeyRateLimit>,
}

fn default_models() -> Vec<String> {
    vec!["*".to_string()]
}

#[derive(Serialize)]
struct KeyResponse {
    name: String,
    key: String,
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate_limit: Option<KeyRateLimit>,
}

#[derive(Serialize)]
struct KeySummary {
    name: String,
    key_prefix: String,
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate_limit: Option<KeyRateLimit>,
    source: &'static str,
}

fn mask_key(key: &str) -> String {
    let prefix: String = key.chars().take(8).collect();
    if prefix.len() < key.len() {
        format!("{prefix}...")
    } else {
        "***".to_string()
    }
}

fn generate_key() -> String {
    use rand::Rng;
    let bytes: [u8; 32] = rand::rng().random();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("sk-{hex}")
}

/// POST /v1/admin/keys — create a new virtual key.
async fn create_key(
    State(state): State<KeyAdminState>,
    Json(body): Json<CreateKeyRequest>,
) -> Response {
    if body.name.is_empty() {
        return err_response(
            StatusCode::BAD_REQUEST,
            "name is required",
            "invalid_request_error",
        );
    }

    // Reject names that collide with TOML-managed keys.
    if state.toml_key_names.contains(&body.name) {
        return err_response(
            StatusCode::CONFLICT,
            &format!("key '{}' is managed by config file", body.name),
            "invalid_request_error",
        );
    }

    // Check storage for existing name (storage is keyed by name, the
    // authoritative source for dynamic keys).
    let skey = storage_key(&PREFIX_KEYS, body.name.as_bytes());
    match state.storage.get(&skey).await {
        Ok(Some(_)) => {
            return err_response(
                StatusCode::CONFLICT,
                &format!("key '{}' already exists", body.name),
                "invalid_request_error",
            );
        }
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
        Ok(None) => {}
    }

    let key = generate_key();
    let config = KeyConfig {
        name: body.name.clone(),
        key: key.clone(),
        models: body.models.clone(),
        rate_limit: body.rate_limit.clone(),
    };

    // Storage-first: persist before updating key_map.
    let value = match serde_json::to_vec(&config) {
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

    // Brief lock — no await while held.
    state
        .key_map
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.clone(), body.name.clone());

    (
        StatusCode::CREATED,
        Json(KeyResponse {
            name: body.name,
            key,
            models: body.models,
            rate_limit: body.rate_limit,
        }),
    )
        .into_response()
}

/// GET /v1/admin/keys — list all virtual keys (TOML + dynamic).
async fn list_keys(State(state): State<KeyAdminState>) -> Response {
    let mut keys: Vec<KeySummary> = state
        .toml_keys
        .iter()
        .map(|kc| KeySummary {
            name: kc.name.clone(),
            key_prefix: mask_key(&kc.key),
            models: kc.models.clone(),
            rate_limit: kc.rate_limit.clone(),
            source: "config",
        })
        .collect();

    let pairs = match state.storage.list(&PREFIX_KEYS).await {
        Ok(p) => p,
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
    };

    for (_k, v) in pairs {
        if let Ok(kc) = serde_json::from_slice::<KeyConfig>(&v) {
            // Skip storage keys that overlap with TOML (TOML already listed).
            if state.toml_key_names.contains(&kc.name) {
                continue;
            }
            keys.push(KeySummary {
                name: kc.name,
                key_prefix: mask_key(&kc.key),
                models: kc.models,
                rate_limit: kc.rate_limit,
                source: "dynamic",
            });
        }
    }

    Json(keys).into_response()
}

/// GET /v1/admin/keys/:name — get a single key's details.
async fn get_key(State(state): State<KeyAdminState>, Path(name): Path<String>) -> Response {
    // Check TOML keys first.
    if let Some(kc) = state.toml_keys.iter().find(|k| k.name == name) {
        return Json(KeySummary {
            name: kc.name.clone(),
            key_prefix: mask_key(&kc.key),
            models: kc.models.clone(),
            rate_limit: kc.rate_limit.clone(),
            source: "config",
        })
        .into_response();
    }

    let skey = storage_key(&PREFIX_KEYS, name.as_bytes());
    match state.storage.get(&skey).await {
        Ok(Some(bytes)) => match serde_json::from_slice::<KeyConfig>(&bytes) {
            Ok(kc) => Json(KeySummary {
                name: kc.name,
                key_prefix: mask_key(&kc.key),
                models: kc.models,
                rate_limit: kc.rate_limit,
                source: "dynamic",
            })
            .into_response(),
            Err(e) => err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            ),
        },
        Ok(None) => err_response(
            StatusCode::NOT_FOUND,
            &format!("key '{name}' not found"),
            "invalid_request_error",
        ),
        Err(e) => err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        ),
    }
}

/// PATCH /v1/admin/keys/:name — partial update of a dynamic key.
///
/// JSON Merge Patch semantics: present fields are set, absent fields
/// are unchanged, `null` clears the field. The `name` and `key` fields
/// are immutable (not accepted in the patch body).
async fn update_key(
    State(state): State<KeyAdminState>,
    Path(name): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if state.toml_key_names.contains(&name) {
        return err_response(
            StatusCode::FORBIDDEN,
            &format!("key '{name}' is managed by config file and cannot be updated via API"),
            "invalid_request_error",
        );
    }

    let skey = storage_key(&PREFIX_KEYS, name.as_bytes());

    // Load the existing config from storage.
    let mut config = match state.storage.get(&skey).await {
        Ok(Some(bytes)) => match serde_json::from_slice::<KeyConfig>(&bytes) {
            Ok(kc) => kc,
            Err(_) => {
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "corrupt key data",
                    "server_error",
                );
            }
        },
        Ok(None) => {
            return err_response(
                StatusCode::NOT_FOUND,
                &format!("key '{name}' not found"),
                "invalid_request_error",
            );
        }
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
    };

    // Reject immutable fields.
    if body.get("name").is_some() || body.get("key").is_some() {
        return err_response(
            StatusCode::BAD_REQUEST,
            "'name' and 'key' are immutable and cannot be patched",
            "invalid_request_error",
        );
    }

    // Apply patch fields.
    if let Some(models) = body.get("models") {
        match serde_json::from_value::<Vec<String>>(models.clone()) {
            Ok(m) => config.models = m,
            Err(e) => {
                return err_response(
                    StatusCode::BAD_REQUEST,
                    &format!("invalid 'models': {e}"),
                    "invalid_request_error",
                );
            }
        }
    }

    if body.get("rate_limit").is_some() {
        match serde_json::from_value::<Option<KeyRateLimit>>(body["rate_limit"].clone()) {
            Ok(rl) => config.rate_limit = rl,
            Err(e) => {
                return err_response(
                    StatusCode::BAD_REQUEST,
                    &format!("invalid 'rate_limit': {e}"),
                    "invalid_request_error",
                );
            }
        }
    }

    // Persist updated config.
    let value = match serde_json::to_vec(&config) {
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

    Json(KeySummary {
        name: config.name,
        key_prefix: mask_key(&config.key),
        models: config.models,
        rate_limit: config.rate_limit,
        source: "dynamic",
    })
    .into_response()
}

/// DELETE /v1/admin/keys/:name — revoke a virtual key.
async fn delete_key(State(state): State<KeyAdminState>, Path(name): Path<String>) -> Response {
    // TOML-managed keys cannot be deleted via the API.
    if state.toml_key_names.contains(&name) {
        return err_response(
            StatusCode::FORBIDDEN,
            &format!("key '{name}' is managed by config file and cannot be deleted via API"),
            "invalid_request_error",
        );
    }

    let skey = storage_key(&PREFIX_KEYS, name.as_bytes());

    // Load the key to find the token for key_map removal.
    let token = match state.storage.get(&skey).await {
        Ok(Some(bytes)) => match serde_json::from_slice::<KeyConfig>(&bytes) {
            Ok(kc) => kc.key,
            Err(_) => {
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "corrupt key data",
                    "server_error",
                );
            }
        },
        Ok(None) => {
            return err_response(
                StatusCode::NOT_FOUND,
                &format!("key '{name}' not found"),
                "invalid_request_error",
            );
        }
        Err(e) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
    };

    // Storage-first: delete from storage before updating key_map.
    if let Err(e) = state.storage.delete(&skey).await {
        return err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        );
    }

    state
        .key_map
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&token);

    StatusCode::NO_CONTENT.into_response()
}

/// Load keys from storage and merge with TOML config keys.
/// TOML keys take precedence on name conflicts.
pub async fn load_stored_keys(
    storage: &dyn Storage,
    toml_keys: &[KeyConfig],
    key_map: &RwLock<HashMap<String, String>>,
) {
    let pairs = match storage.list(&PREFIX_KEYS).await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: failed to load stored keys: {e}");
            return;
        }
    };

    let toml_names: HashSet<&str> = toml_keys.iter().map(|k| k.name.as_str()).collect();

    let mut map = key_map.write().unwrap_or_else(|e| e.into_inner());
    for (_k, v) in pairs {
        if let Ok(kc) = serde_json::from_slice::<KeyConfig>(&v) {
            // TOML keys take precedence — skip storage keys that conflict.
            if toml_names.contains(kc.name.as_str()) {
                continue;
            }
            map.insert(kc.key, kc.name);
        }
    }
}

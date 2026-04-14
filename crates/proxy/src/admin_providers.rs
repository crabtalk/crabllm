use crate::PREFIX_PROVIDERS;
use arc_swap::ArcSwap;
use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use crabllm_core::{
    Error, GatewayConfig, Provider, ProviderConfig, ProviderKind, Storage, storage_key,
};
use crabllm_provider::{HttpClient, ProviderRegistry};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

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
    storage: Arc<dyn Storage>,
    /// Serializes mutation paths (create/update/delete/reload). Read paths
    /// (list/get) don't acquire this — they only touch storage + the
    /// config file, both of which are fine under concurrent access.
    write_lock: Arc<Mutex<()>>,
}

impl<P: Provider> Clone for ProviderAdminState<P> {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            config_path: self.config_path.clone(),
            admin_token: self.admin_token.clone(),
            rebuilder: self.rebuilder.clone(),
            storage: self.storage.clone(),
            write_lock: self.write_lock.clone(),
        }
    }
}

/// Build admin provider management routes, protected by admin token auth.
pub fn provider_admin_routes<P: Provider + 'static>(
    registry: Arc<ArcSwap<ProviderRegistry<P>>>,
    config_path: PathBuf,
    admin_token: String,
    rebuilder: Rebuilder<P>,
    storage: Arc<dyn Storage>,
) -> Router {
    let state = ProviderAdminState {
        registry,
        config_path,
        admin_token,
        rebuilder,
        storage,
        write_lock: Arc::new(Mutex::new(())),
    };
    Router::new()
        .route(
            "/v1/admin/providers",
            post(create_provider::<P>).get(list_providers::<P>),
        )
        .route(
            "/v1/admin/providers/{name}",
            get(get_provider::<P>)
                .patch(update_provider::<P>)
                .delete(delete_provider::<P>),
        )
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
    if let Err(r) = crate::admin::check_admin_token(&request, &state.admin_token) {
        return r;
    }
    next.run(request).await
}

// ── CRUD ──

/// Request body for `POST /v1/admin/providers`. Flat shape: a
/// provider name plus the full `ProviderConfig` inline.
#[derive(Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub(crate) struct CreateProviderRequest {
    name: String,
    #[serde(default, alias = "standard")]
    kind: ProviderKind,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    weight: Option<u16>,
    #[serde(default)]
    max_retries: Option<u32>,
    #[serde(default)]
    api_version: Option<String>,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default)]
    region: Option<String>,
    #[serde(default)]
    access_key: Option<String>,
    #[serde(default)]
    secret_key: Option<String>,
}

impl CreateProviderRequest {
    fn into_parts(self) -> (String, ProviderConfig) {
        (
            self.name,
            ProviderConfig {
                kind: self.kind,
                api_key: self.api_key,
                base_url: self.base_url,
                models: self.models,
                weight: self.weight,
                max_retries: self.max_retries,
                api_version: self.api_version,
                timeout: self.timeout,
                region: self.region,
                access_key: self.access_key,
                secret_key: self.secret_key,
            },
        )
    }
}

/// Response shape for provider GET — secrets masked.
#[derive(Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub(crate) struct ProviderSummary {
    name: String,
    kind: ProviderKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key_prefix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_key_prefix: Option<String>,
    source: &'static str,
}

fn summarize(name: &str, cfg: &ProviderConfig, source: &'static str) -> ProviderSummary {
    ProviderSummary {
        name: name.to_string(),
        kind: cfg.kind.clone(),
        api_key_prefix: cfg.api_key.as_deref().map(mask),
        base_url: cfg.base_url.clone(),
        models: cfg.models.clone(),
        weight: cfg.weight,
        max_retries: cfg.max_retries,
        api_version: cfg.api_version.clone(),
        timeout: cfg.timeout,
        region: cfg.region.clone(),
        access_key_prefix: cfg.access_key.as_deref().map(mask),
        source,
    }
}

fn mask(secret: &str) -> String {
    let prefix: String = secret.chars().take(8).collect();
    if prefix.len() < secret.len() {
        format!("{prefix}...")
    } else {
        "***".to_string()
    }
}

/// GET /v1/admin/providers — list TOML + dynamic providers. Secrets masked.
async fn list_providers<P: Provider>(State(state): State<ProviderAdminState<P>>) -> Response {
    let toml_config = match read_toml_config(&state.config_path).await {
        Ok(c) => c,
        Err(r) => return r,
    };
    let toml_names: HashSet<String> = toml_config.providers.keys().cloned().collect();

    let mut summaries: Vec<ProviderSummary> = toml_config
        .providers
        .iter()
        .map(|(name, cfg)| summarize(name, cfg, "config"))
        .collect();

    let pairs = match state.storage.list(&PREFIX_PROVIDERS).await {
        Ok(p) => p,
        Err(e) => {
            return crate::admin::err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
    };

    for (_k, v) in pairs {
        let Ok(cfg) = serde_json::from_slice::<StoredProvider>(&v) else {
            continue;
        };
        if toml_names.contains(&cfg.name) {
            continue;
        }
        summaries.push(summarize(&cfg.name, &cfg.config, "dynamic"));
    }

    Json(summaries).into_response()
}

/// GET /v1/admin/providers/{name} — get one provider.
async fn get_provider<P: Provider>(
    State(state): State<ProviderAdminState<P>>,
    Path(name): Path<String>,
) -> Response {
    let toml_config = match read_toml_config(&state.config_path).await {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Some(cfg) = toml_config.providers.get(&name) {
        return Json(summarize(&name, cfg, "config")).into_response();
    }

    match load_stored(state.storage.as_ref(), &name).await {
        Ok(Some(cfg)) => Json(summarize(&name, &cfg, "dynamic")).into_response(),
        Ok(None) => crate::admin::err_response(
            StatusCode::NOT_FOUND,
            &format!("provider '{name}' not found"),
            "invalid_request_error",
        ),
        Err(r) => r,
    }
}

/// POST /v1/admin/providers — create a new dynamic provider.
async fn create_provider<P: Provider>(
    State(state): State<ProviderAdminState<P>>,
    Json(body): Json<CreateProviderRequest>,
) -> Response {
    if body.name.is_empty() {
        return crate::admin::err_response(
            StatusCode::BAD_REQUEST,
            "name is required",
            "invalid_request_error",
        );
    }
    let (name, mut config) = body.into_parts();

    let _guard = state.write_lock.lock().await;

    let toml_config = match read_toml_config(&state.config_path).await {
        Ok(c) => c,
        Err(r) => return r,
    };
    if toml_config.providers.contains_key(&name) {
        return crate::admin::err_response(
            StatusCode::CONFLICT,
            &format!("provider '{name}' is managed by config file"),
            "invalid_request_error",
        );
    }

    let skey = storage_key(&PREFIX_PROVIDERS, name.as_bytes());
    match state.storage.get(&skey).await {
        Ok(Some(_)) => {
            return crate::admin::err_response(
                StatusCode::CONFLICT,
                &format!("provider '{name}' already exists"),
                "invalid_request_error",
            );
        }
        Err(e) => {
            return crate::admin::err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
        Ok(None) => {}
    }

    if let Err(e) = autofill_models(&mut config).await {
        return crate::admin::err_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if let Err(e) = validate_single(&name, &config) {
        return crate::admin::err_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if let Err(r) = commit_change(&state, &name, Some(&config)).await {
        return r;
    }

    (
        StatusCode::CREATED,
        Json(summarize(&name, &config, "dynamic")),
    )
        .into_response()
}

/// PATCH /v1/admin/providers/{name} — partial update of a dynamic provider.
async fn update_provider<P: Provider>(
    State(state): State<ProviderAdminState<P>>,
    Path(name): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if body.get("name").is_some() {
        return crate::admin::err_response(
            StatusCode::BAD_REQUEST,
            "'name' is immutable and cannot be patched",
            "invalid_request_error",
        );
    }

    let _guard = state.write_lock.lock().await;

    let toml_config = match read_toml_config(&state.config_path).await {
        Ok(c) => c,
        Err(r) => return r,
    };
    if toml_config.providers.contains_key(&name) {
        return crate::admin::err_response(
            StatusCode::FORBIDDEN,
            &format!("provider '{name}' is managed by config file and cannot be updated via API"),
            "invalid_request_error",
        );
    }

    let mut config = match load_stored(state.storage.as_ref(), &name).await {
        Ok(Some(c)) => c,
        Ok(None) => {
            return crate::admin::err_response(
                StatusCode::NOT_FOUND,
                &format!("provider '{name}' not found"),
                "invalid_request_error",
            );
        }
        Err(r) => return r,
    };

    if let Err(r) = apply_patch(&mut config, &body) {
        return r;
    }

    if let Err(e) = validate_single(&name, &config) {
        return crate::admin::err_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if let Err(r) = commit_change(&state, &name, Some(&config)).await {
        return r;
    }

    Json(summarize(&name, &config, "dynamic")).into_response()
}

/// DELETE /v1/admin/providers/{name} — delete a dynamic provider.
async fn delete_provider<P: Provider>(
    State(state): State<ProviderAdminState<P>>,
    Path(name): Path<String>,
) -> Response {
    let _guard = state.write_lock.lock().await;

    let toml_config = match read_toml_config(&state.config_path).await {
        Ok(c) => c,
        Err(r) => return r,
    };
    if toml_config.providers.contains_key(&name) {
        return crate::admin::err_response(
            StatusCode::FORBIDDEN,
            &format!("provider '{name}' is managed by config file and cannot be deleted via API"),
            "invalid_request_error",
        );
    }

    let skey = storage_key(&PREFIX_PROVIDERS, name.as_bytes());
    match state.storage.get(&skey).await {
        Ok(None) => {
            return crate::admin::err_response(
                StatusCode::NOT_FOUND,
                &format!("provider '{name}' not found"),
                "invalid_request_error",
            );
        }
        Err(e) => {
            return crate::admin::err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            );
        }
        Ok(Some(_)) => {}
    }

    if let Err(r) = commit_change(&state, &name, None).await {
        return r;
    }

    StatusCode::NO_CONTENT.into_response()
}

// ── Helpers ──

/// Storage row: name + config. We store the name inside the value so
/// listing doesn't need to decode raw storage keys.
#[derive(Serialize, Deserialize)]
struct StoredProvider {
    name: String,
    #[serde(flatten)]
    config: ProviderConfig,
}

/// Merge dynamic providers from storage into a config's provider map.
/// TOML providers take precedence on name conflicts. Called at startup,
/// during reload, and after every CRUD mutation before rebuilding.
pub async fn merge_stored_providers(storage: &dyn Storage, config: &mut GatewayConfig) {
    let pairs = match storage.list(&PREFIX_PROVIDERS).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("failed to load stored providers: {e}");
            return;
        }
    };
    for (_k, v) in pairs {
        let Ok(sp) = serde_json::from_slice::<StoredProvider>(&v) else {
            continue;
        };
        // TOML precedence: log a warning and skip if name already present.
        if config.providers.contains_key(&sp.name) {
            tracing::warn!(
                name = %sp.name,
                "dynamic provider shadowed by TOML-managed provider of the same name"
            );
            continue;
        }
        config.providers.insert(sp.name, sp.config);
    }
}

async fn read_toml_config(path: &PathBuf) -> Result<GatewayConfig, Response> {
    let raw = tokio::fs::read_to_string(path).await.map_err(|e| {
        crate::admin::err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to read config file: {e}"),
            "server_error",
        )
    })?;
    toml::from_str::<GatewayConfig>(&raw).map_err(|e| {
        crate::admin::err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to parse config: {e}"),
            "server_error",
        )
    })
}

async fn load_stored(
    storage: &dyn Storage,
    name: &str,
) -> Result<Option<ProviderConfig>, Response> {
    let skey = storage_key(&PREFIX_PROVIDERS, name.as_bytes());
    match storage.get(&skey).await {
        Ok(Some(bytes)) => match serde_json::from_slice::<StoredProvider>(&bytes) {
            Ok(sp) => Ok(Some(sp.config)),
            Err(_) => Err(crate::admin::err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "corrupt provider data",
                "server_error",
            )),
        },
        Ok(None) => Ok(None),
        Err(e) => Err(crate::admin::err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        )),
    }
}

#[allow(clippy::result_large_err)]
fn apply_patch(config: &mut ProviderConfig, body: &serde_json::Value) -> Result<(), Response> {
    let obj = body.as_object().ok_or_else(|| {
        crate::admin::err_response(
            StatusCode::BAD_REQUEST,
            "request body must be a JSON object",
            "invalid_request_error",
        )
    })?;

    for (key, value) in obj {
        match key.as_str() {
            "kind" => {
                config.kind = serde_json::from_value(value.clone()).map_err(|e| {
                    crate::admin::err_response(
                        StatusCode::BAD_REQUEST,
                        &format!("invalid 'kind': {e}"),
                        "invalid_request_error",
                    )
                })?;
            }
            "api_key" => {
                config.api_key = from_value_opt(value, "api_key")?;
            }
            "base_url" => {
                config.base_url = from_value_opt(value, "base_url")?;
            }
            "models" => {
                config.models = serde_json::from_value(value.clone()).map_err(|e| {
                    crate::admin::err_response(
                        StatusCode::BAD_REQUEST,
                        &format!("invalid 'models': {e}"),
                        "invalid_request_error",
                    )
                })?;
            }
            "weight" => config.weight = from_value_opt(value, "weight")?,
            "max_retries" => config.max_retries = from_value_opt(value, "max_retries")?,
            "api_version" => config.api_version = from_value_opt(value, "api_version")?,
            "timeout" => config.timeout = from_value_opt(value, "timeout")?,
            "region" => config.region = from_value_opt(value, "region")?,
            "access_key" => config.access_key = from_value_opt(value, "access_key")?,
            "secret_key" => config.secret_key = from_value_opt(value, "secret_key")?,
            other => {
                return Err(crate::admin::err_response(
                    StatusCode::BAD_REQUEST,
                    &format!("unknown field '{other}'"),
                    "invalid_request_error",
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::result_large_err)]
fn from_value_opt<T: for<'de> Deserialize<'de>>(
    value: &serde_json::Value,
    field: &str,
) -> Result<Option<T>, Response> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|e| {
            crate::admin::err_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid '{field}': {e}"),
                "invalid_request_error",
            )
        })
}

fn validate_single(name: &str, config: &ProviderConfig) -> Result<(), String> {
    config.validate(name)
}

/// If `config.models` is empty, query the provider's `GET {base_url}/models`
/// and populate from the response. Only OpenAI-compatible kinds expose a
/// standard models endpoint — other kinds error out asking for an explicit
/// `--models`.
async fn autofill_models(config: &mut ProviderConfig) -> Result<(), String> {
    if !config.models.is_empty() {
        return Ok(());
    }

    let base_url = match &config.kind {
        crabllm_core::ProviderKind::Openai => config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1"),
        crabllm_core::ProviderKind::Ollama => config
            .base_url
            .as_deref()
            .unwrap_or("http://localhost:11434/v1"),
        crabllm_core::ProviderKind::Custom(_) => config.base_url.as_deref().ok_or_else(|| {
            "models is empty and base_url is not set; cannot auto-fetch".to_string()
        })?,
        other => {
            return Err(format!(
                "models is required for kind '{other}' — auto-fetch only supported for \
                 openai, ollama, and custom kinds"
            ));
        }
    };

    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let auth = config.api_key.as_ref().map(|k| format!("Bearer {k}"));
    let mut headers: Vec<(&str, &str)> = Vec::new();
    if let Some(h) = auth.as_deref() {
        headers.push(("authorization", h));
    }

    let client = HttpClient::new();
    let resp = client
        .get(&url, &headers)
        .await
        .map_err(|e| format!("failed to auto-fetch models from {url}: {e}"))?;

    if !(200..300).contains(&resp.status) {
        return Err(format!(
            "{url} returned {}; pass --models explicitly",
            resp.status
        ));
    }

    let body: serde_json::Value =
        serde_json::from_slice(&resp.body).map_err(|e| format!("invalid JSON from {url}: {e}"))?;
    let data = body
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{url} missing 'data' array; pass --models explicitly"))?;

    let models: Vec<String> = data
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(String::from))
        .collect();

    if models.is_empty() {
        return Err(format!(
            "{url} returned no models; pass --models explicitly"
        ));
    }

    tracing::info!(
        kind = %config.kind,
        base_url,
        count = models.len(),
        "auto-fetched models from provider",
    );

    config.models = models;
    Ok(())
}

/// Apply a single-provider mutation: build registry first (on a projected
/// config), persist to storage only if build succeeds, then swap. This
/// ordering guarantees that a rebuild failure never leaves a corrupted
/// row behind in storage.
///
/// `new_config = Some(_)` means create/update; `None` means delete.
/// Caller must hold `state.write_lock` to serialize mutations.
#[allow(clippy::result_large_err)]
async fn commit_change<P: Provider>(
    state: &ProviderAdminState<P>,
    name: &str,
    new_config: Option<&ProviderConfig>,
) -> Result<(), Response> {
    let mut config = read_toml_config(&state.config_path).await?;
    merge_stored_providers(state.storage.as_ref(), &mut config).await;
    match new_config {
        Some(c) => {
            config.providers.insert(name.to_string(), c.clone());
        }
        None => {
            config.providers.remove(name);
        }
    }

    // Build first — no side effects on failure.
    let new_registry = (state.rebuilder)(&config).map_err(|e| {
        crate::admin::err_response(
            StatusCode::BAD_REQUEST,
            &format!("failed to rebuild registry: {e}"),
            "invalid_request_error",
        )
    })?;

    // Persist second — storage matches the registry we're about to swap in.
    let skey = storage_key(&PREFIX_PROVIDERS, name.as_bytes());
    match new_config {
        Some(c) => {
            let stored = StoredProvider {
                name: name.to_string(),
                config: c.clone(),
            };
            let value = serde_json::to_vec(&stored).map_err(|e| {
                crate::admin::err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                )
            })?;
            state.storage.set(&skey, value).await.map_err(|e| {
                crate::admin::err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                )
            })?;
        }
        None => {
            state.storage.delete(&skey).await.map_err(|e| {
                crate::admin::err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                )
            })?;
        }
    }

    // Swap last — infallible.
    state.registry.store(Arc::new(new_registry));
    Ok(())
}

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level gateway configuration, loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Address to listen on, e.g. "0.0.0.0:8080".
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Named provider configurations.
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    /// Virtual API keys for client authentication.
    #[serde(default)]
    pub keys: Vec<KeyConfig>,
    /// Extension configurations. Each key is an extension name, value is its config.
    #[serde(default)]
    pub extensions: Option<serde_json::Value>,
    /// Storage backend configuration.
    #[serde(default)]
    pub storage: Option<StorageConfig>,
    /// Model name aliases. Maps friendly names to canonical model names.
    #[serde(default)]
    pub aliases: HashMap<String, String>,
    /// Per-model metadata overrides (context window, pricing). Merged with
    /// built-in defaults at lookup time — only specify what you want to override.
    #[serde(default)]
    pub models: HashMap<String, crate::ModelInfo>,
    /// Path to cloud model metadata TOML file (pricing + context windows).
    /// Entries are merged into `models` at startup (config entries win).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloud_models: Option<String>,
    /// Admin API bearer token. If set, enables /v1/admin/* endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admin_token: Option<String>,
    /// Graceful shutdown timeout in seconds. Default: 30.
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: u64,
    /// Serve OpenAPI documentation at `/openapi.json` and `/docs`.
    /// Defaults to `true`; set `openapi = false` to disable.
    /// Ignored unless the binary is built with the `openapi` feature.
    #[serde(default = "default_openapi")]
    pub openapi: bool,
}

/// Configuration for a single LLM provider.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider kind determines the dispatch path. When omitted, the
    /// provider's map key doubles as the kind — `[providers.zai]` with no
    /// `kind` resolves to kind `zai`. Set it explicitly only when the
    /// instance name differs from the kind (e.g. two `openai` instances).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ProviderKind>,
    /// API key (supports `${ENV_VAR}` interpolation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL override. OpenAI-compat providers have sensible defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Model names served by this provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// Routing weight for weighted random selection. Higher = more traffic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u16>,
    /// Max retries on transient errors before fallback. 0 disables retry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// API version string, used by Azure OpenAI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    /// Per-request timeout in seconds. Default: 30.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    /// Wall-clock deadline for the entire retry loop in seconds. Default: 15.
    /// The loop stops retrying once this much time has elapsed since the
    /// first attempt, even if `max_retries` has not been exhausted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_deadline: Option<u64>,
}

fn default_shutdown_timeout() -> u64 {
    30
}

fn default_openapi() -> bool {
    true
}

fn default_listen() -> String {
    "127.0.0.1:5632".to_string()
}

/// Which provider implementation to use. Known variants map to named
/// dispatch paths; a self-defined name deserializes to [`Custom`] — see its
/// note for how it resolves (compat-table entry vs. bare OpenAI-compatible).
///
/// [`Custom`]: ProviderKind::Custom
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ProviderKind {
    #[default]
    Openai,
    Anthropic,
    Google,
    Ollama,
    Azure,
    /// Self-defined kind — any string that doesn't match a known variant.
    /// Dispatched through the OpenAI-compatible path, unless it names a
    /// built-in OpenAI+Anthropic provider (see the provider crate's compat
    /// table), in which case its endpoints come from that table.
    Custom(String),
}

impl ProviderKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::Ollama => "ollama",
            Self::Azure => "azure",
            Self::Custom(s) => s,
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for ProviderKind {
    fn from(s: String) -> Self {
        match s.as_str() {
            "openai" => Self::Openai,
            "anthropic" => Self::Anthropic,
            "google" => Self::Google,
            "ollama" => Self::Ollama,
            "azure" => Self::Azure,
            _ => Self::Custom(s),
        }
    }
}

impl serde::Serialize for ProviderKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for ProviderKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self::from(String::deserialize(d)?))
    }
}

impl ProviderConfig {
    /// Resolve the effective provider kind. When `kind` is omitted, the
    /// provider's map key (`name`) is used as the kind string. As a final
    /// nudge, an OpenAI kind whose `base_url` points at an Anthropic
    /// endpoint resolves to `Anthropic`.
    pub fn effective_kind(&self, name: &str) -> ProviderKind {
        let kind = self
            .kind
            .clone()
            .unwrap_or_else(|| ProviderKind::from(name.to_string()));
        if kind == ProviderKind::Openai
            && self
                .base_url
                .as_deref()
                .is_some_and(|url| url.contains("anthropic"))
        {
            return ProviderKind::Anthropic;
        }
        kind
    }
}

/// Per-key rate limit override. When set on a key, these values take
/// precedence over the global `[extensions.rate_limit]` config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct KeyRateLimit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_minute: Option<u64>,
}

/// Virtual API key for client authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyConfig {
    /// Human-readable name for this key.
    pub name: String,
    /// The key string clients send in Authorization header.
    pub key: String,
    /// Which models this key can access. `["*"]` means all.
    pub models: Vec<String>,
    /// Per-key rate limit override. Takes precedence over the global
    /// `[extensions.rate_limit]` config when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<KeyRateLimit>,
}

/// Storage backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Backend kind: "memory" (default) or "sqlite" (requires feature).
    #[serde(default = "StorageConfig::default_kind")]
    pub kind: String,
    /// File path for persistent backends (required for sqlite).
    #[serde(default)]
    pub path: Option<String>,
}

impl StorageConfig {
    fn default_kind() -> String {
        "memory".to_string()
    }
}

impl GatewayConfig {
    /// Load config from a TOML file, expanding `${VAR}` patterns in
    /// string values. If `cloud_models` is set, loads the referenced
    /// file and merges entries into `models` (config entries win over
    /// cloud file entries).
    #[cfg(feature = "gateway")]
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let expanded = expand_env_vars(&raw);

        let mut config: GatewayConfig = toml::from_str(&expanded)?;

        let config_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        config.load_cloud_models(config_dir)?;

        Ok(config)
    }

    /// Load cloud model metadata from the configured TOML file and merge
    /// into `self.models`. Config entries take precedence — cloud file
    /// entries only fill gaps.
    #[cfg(feature = "gateway")]
    fn load_cloud_models(
        &mut self,
        config_dir: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(ref path) = self.cloud_models else {
            return Ok(());
        };
        let full = config_dir.join(path);
        let raw = std::fs::read_to_string(&full)
            .map_err(|e| format!("cloud_models '{}': {e}", full.display()))?;
        let table: HashMap<String, crate::ModelInfo> =
            toml::from_str(&raw).map_err(|e| format!("cloud_models '{}': {e}", full.display()))?;
        for (model, info) in table {
            self.models.entry(model).or_insert(info);
        }
        Ok(())
    }
}

/// Expand `${VAR}` patterns in a string using environment variables.
/// Unknown variables are replaced with empty string.
#[cfg(feature = "gateway")]
fn expand_env_vars(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_name = String::new();
            for ch in chars.by_ref() {
                if ch == '}' {
                    break;
                }
                var_name.push(ch);
            }
            if let Ok(val) = std::env::var(&var_name) {
                result.push_str(&val);
            }
        } else {
            result.push(c);
        }
    }

    result
}

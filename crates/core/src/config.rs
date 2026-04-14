use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-model token pricing configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PricingConfig {
    /// Cost per million prompt tokens in USD.
    pub prompt_cost_per_million: f64,
    /// Cost per million completion tokens in USD.
    pub completion_cost_per_million: f64,
}

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
    /// Path to local model registry TOML file (alias → HF repo ID).
    /// Extends the build-time MLX model registry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_models: Option<String>,
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
    /// Provider kind determines the dispatch path.
    #[serde(default, skip_serializing_if = "ProviderKind::is_default")]
    pub kind: ProviderKind,
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
    /// AWS region for Bedrock provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// AWS access key ID for Bedrock provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_key: Option<String>,
    /// AWS secret access key for Bedrock provider.
    #[serde(default, skip_serializing)]
    pub secret_key: Option<String>,
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
/// dispatch paths. A self-defined name deserializes to [`Custom`], which
/// dispatches as OpenAI-compatible and requires `base_url` at validation.
///
/// [`Custom`]: ProviderKind::Custom
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ProviderKind {
    #[default]
    Openai,
    Anthropic,
    Google,
    Bedrock,
    Ollama,
    Azure,
    /// Self-defined kind — any string that doesn't match a known variant.
    /// Dispatched through the OpenAI-compatible path; `base_url` required.
    Custom(String),
}

impl ProviderKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::Bedrock => "bedrock",
            Self::Ollama => "ollama",
            Self::Azure => "azure",
            Self::Custom(s) => s,
        }
    }

    /// Returns true if this is the default variant (Openai).
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Openai)
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl serde::Serialize for ProviderKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for ProviderKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "openai" => Self::Openai,
            "anthropic" => Self::Anthropic,
            "google" => Self::Google,
            "bedrock" => Self::Bedrock,
            "ollama" => Self::Ollama,
            "azure" => Self::Azure,
            _ => Self::Custom(s),
        })
    }
}

impl ProviderConfig {
    /// Resolve the effective provider kind.
    ///
    /// Returns `Anthropic` if the field is explicitly set to `Anthropic`,
    /// or if `base_url` contains "anthropic". Otherwise returns the
    /// configured kind.
    pub fn effective_kind(&self) -> ProviderKind {
        if self.kind == ProviderKind::Anthropic {
            return ProviderKind::Anthropic;
        }
        if let Some(url) = &self.base_url
            && url.contains("anthropic")
        {
            return ProviderKind::Anthropic;
        }
        self.kind.clone()
    }

    /// Validate field combinations.
    pub fn validate(&self, provider_name: &str) -> Result<(), String> {
        if self.models.is_empty() {
            return Err(format!("provider '{provider_name}' has no models"));
        }
        match &self.kind {
            ProviderKind::Bedrock => {
                if self.region.is_none() {
                    return Err(format!(
                        "provider '{provider_name}' (bedrock) requires region"
                    ));
                }
                if self.access_key.is_none() {
                    return Err(format!(
                        "provider '{provider_name}' (bedrock) requires access_key"
                    ));
                }
                if self.secret_key.is_none() {
                    return Err(format!(
                        "provider '{provider_name}' (bedrock) requires secret_key"
                    ));
                }
            }
            ProviderKind::Ollama => {
                // Ollama doesn't require api_key or base_url.
            }
            ProviderKind::Custom(name) => {
                if self.base_url.is_none() {
                    return Err(format!(
                        "provider '{provider_name}' (custom kind '{name}') requires base_url"
                    ));
                }
            }
            _ => {
                if self.api_key.is_none() && self.base_url.is_none() {
                    return Err(format!(
                        "provider '{provider_name}' requires api_key or base_url"
                    ));
                }
            }
        }
        Ok(())
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

/// A single local model entry: HF repo ID and optional disk size.
#[derive(Debug, Clone, Deserialize)]
pub struct LocalModelEntry {
    pub repo_id: String,
    /// Approximate disk size in megabytes.
    #[serde(default)]
    pub size_mb: Option<u64>,
    /// Whether the model accepts image/video input (VLM).
    #[serde(default)]
    pub vision: Option<bool>,
    /// Architecture / model_type from HuggingFace config (e.g. `"qwen2"`, `"llama"`).
    #[serde(default)]
    pub arch: Option<String>,
}

/// Wrapper for local model TOML: `[models.family.size.quant]` nested tables.
#[cfg(feature = "gateway")]
#[derive(Deserialize)]
struct LocalModelsFile {
    #[serde(default)]
    models: HashMap<String, HashMap<String, HashMap<String, LocalModelEntry>>>,
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

    /// Load local model entries from the configured TOML file.
    ///
    /// The TOML uses nested tables `[models.family.size.quant]`. This
    /// flattens them to `"family.size.quant" → entry` for consumers.
    #[cfg(feature = "gateway")]
    pub fn load_local_models(
        &self,
        config_dir: &std::path::Path,
    ) -> Result<HashMap<String, LocalModelEntry>, Box<dyn std::error::Error>> {
        let Some(ref path) = self.local_models else {
            return Ok(HashMap::new());
        };
        let full = config_dir.join(path);
        let raw = std::fs::read_to_string(&full)
            .map_err(|e| format!("local_models '{}': {e}", full.display()))?;
        let file: LocalModelsFile =
            toml::from_str(&raw).map_err(|e| format!("local_models '{}': {e}", full.display()))?;

        let mut result = HashMap::new();
        for (family, sizes) in file.models {
            for (size, quants) in sizes {
                for (quant, entry) in quants {
                    let alias = format!("{family}.{size}.{quant}");
                    result.insert(alias, entry);
                }
            }
        }
        Ok(result)
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

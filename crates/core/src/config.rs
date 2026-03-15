use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, path::Path};

/// Top-level gateway configuration, loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Address to listen on, e.g. "0.0.0.0:8080".
    pub listen: String,
    /// Named provider configurations.
    pub providers: HashMap<String, ProviderConfig>,
    /// Model name → provider routing.
    pub models: HashMap<String, ModelRoute>,
    /// Virtual API keys for client authentication.
    #[serde(default)]
    pub keys: Vec<KeyConfig>,
    /// Extension configurations. Each key is an extension name, value is its config.
    #[serde(default)]
    pub extensions: Option<toml::Value>,
    /// Storage backend configuration.
    #[serde(default)]
    pub storage: Option<StorageConfig>,
}

/// Configuration for a single LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider kind determines the dispatch path.
    pub kind: ProviderKind,
    /// API key (supports `${ENV_VAR}` interpolation).
    #[serde(default)]
    pub api_key: String,
    /// Base URL override. OpenAI-compat providers have sensible defaults.
    #[serde(default)]
    pub base_url: Option<String>,
}

/// Which provider implementation to use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenaiCompat,
    Anthropic,
    Google,
    Bedrock,
}

/// Maps a model name to a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRoute {
    /// Name of the provider in the `providers` table.
    pub provider: String,
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
}

/// Storage backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Backend kind: "memory" (default) or "sqlite" (requires feature).
    #[serde(default = "StorageConfig::default_kind")]
    pub kind: String,
}

impl StorageConfig {
    fn default_kind() -> String {
        "memory".to_string()
    }
}

impl GatewayConfig {
    /// Load config from a TOML file, expanding `${VAR}` patterns in string values.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let expanded = expand_env_vars(&raw);
        let config: GatewayConfig = toml::from_str(&expanded)?;
        Ok(config)
    }
}

/// Expand `${VAR}` patterns in a string using environment variables.
/// Unknown variables are replaced with empty string.
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
            if let Ok(val) = env::var(&var_name) {
                result.push_str(&val);
            }
        } else {
            result.push(c);
        }
    }

    result
}

use serde::{Deserialize, Serialize};

/// Per-key rate limit (mirrors server-side KeyRateLimit).
#[derive(Debug, Deserialize, Serialize)]
pub struct KeyRateLimit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_minute: Option<u64>,
}

/// POST /v1/admin/keys request body.
#[derive(Serialize)]
pub struct CreateKeyRequest {
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<KeyRateLimit>,
}

/// POST /v1/admin/keys response (full key visible once).
#[derive(Deserialize, Serialize)]
pub struct KeyResponse {
    pub name: String,
    pub key: String,
    pub models: Vec<String>,
    pub rate_limit: Option<KeyRateLimit>,
}

/// Format rate limit fields as (rpm, tpm) display strings.
pub fn format_rate_limit(rl: &Option<KeyRateLimit>) -> (String, String) {
    let rpm = rl
        .as_ref()
        .and_then(|r| r.requests_per_minute)
        .map_or("-".into(), |v| v.to_string());
    let tpm = rl
        .as_ref()
        .and_then(|r| r.tokens_per_minute)
        .map_or("-".into(), |v| v.to_string());
    (rpm, tpm)
}

/// GET /v1/admin/keys and GET /v1/admin/keys/{name} response.
#[derive(Deserialize, Serialize)]
pub struct KeySummary {
    pub name: String,
    pub key_prefix: String,
    pub models: Vec<String>,
    pub rate_limit: Option<KeyRateLimit>,
    pub source: String,
}

/// GET /v1/admin/usage entry.
#[derive(Deserialize, Serialize)]
pub struct UsageEntry {
    pub name: String,
    pub model: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
}

/// GET /v1/budget entry.
#[derive(Deserialize, Serialize)]
pub struct BudgetEntry {
    pub key: String,
    pub spent_usd: f64,
    pub budget_usd: f64,
    pub remaining_usd: f64,
}

/// Provider implementation kind — mirrors server-side `ProviderKind`.
/// Known kinds (openai, anthropic, google, bedrock, ollama, azure) pick
/// named dispatch paths; any other string is treated by the server as a
/// self-defined OpenAI-compatible kind and requires `base_url`.
pub type ProviderKind = String;

/// POST /v1/admin/providers request body.
#[derive(Serialize)]
pub struct CreateProviderRequest {
    pub name: String,
    pub kind: ProviderKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_key: Option<String>,
}

/// GET /v1/admin/providers response (secrets masked).
#[derive(Deserialize, Serialize)]
pub struct ProviderSummary {
    pub name: String,
    pub kind: ProviderKind,
    pub api_key_prefix: Option<String>,
    pub base_url: Option<String>,
    pub models: Vec<String>,
    pub weight: Option<u16>,
    pub max_retries: Option<u32>,
    pub api_version: Option<String>,
    pub timeout: Option<u64>,
    pub region: Option<String>,
    pub access_key_prefix: Option<String>,
    pub source: String,
}

/// GET /v1/admin/logs entry.
#[derive(Deserialize, Serialize)]
pub struct AuditRecord {
    pub request_id: String,
    pub timestamp: i64,
    pub principal: String,
    pub model: String,
    pub provider: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub cost_micros: i64,
    pub latency_ms: u64,
    pub status: u16,
}

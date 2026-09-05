use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
use serde::{Deserialize, Serialize};

/// A wire dialect a model can be addressed in — i.e. which gateway endpoint
/// natively serves it. Reported per model by `/v1/models` so clients can pick
/// the right endpoint (or translate) without knowing the server's routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum Dialect {
    Openai,
    Anthropic,
    Gemini,
}

/// One model, in the shape the gateway emits. The defaults are for the other
/// direction: `id` is all a real provider is guaranteed to send — DeepSeek
/// omits `created`, and Anthropic's list carries neither it nor `owned_by` —
/// and nothing routes on the rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Model {
    pub id: String,
    #[serde(default, deserialize_with = "lenient")]
    pub object: String,
    #[serde(default, deserialize_with = "lenient")]
    pub created: u64,
    #[serde(default, deserialize_with = "lenient")]
    pub owned_by: String,
    #[serde(
        default,
        deserialize_with = "lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub context_length: Option<u32>,
    #[serde(
        default,
        deserialize_with = "lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub pricing: Option<PricingConfig>,
    #[serde(
        default,
        deserialize_with = "lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub vision: Option<bool>,
    /// Native dialects for this model — the endpoints that serve it without
    /// translation. Empty is omitted for backward compatibility.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dialects: Vec<Dialect>,
}

/// Every field but `id` is best-effort: a third party's row is its own schema,
/// so it may arrive as `null`, or as a type we never asked for — OpenRouter
/// quotes `pricing` as per-token strings where this is per-million floats.
/// Take what fits and default the rest; `#[serde(default)]` alone covers only
/// an absent key, and one odd field must not fail the whole catalog.
fn lenient<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned + Default,
{
    let value = serde_json::Value::deserialize(d)?;
    Ok(serde_json::from_value(value).unwrap_or_default())
}

impl Model {
    /// A model known only by its id — what a provider's list endpoint leaves
    /// once its own dialect is translated away.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: 0,
            owned_by: String::new(),
            context_length: None,
            pricing: None,
            vision: None,
            dialects: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ModelList {
    #[serde(default)]
    pub object: String,
    pub data: Vec<Model>,
}

/// Per-model token pricing. One rate per usage axis. Secondary rates are
/// `Option<f64>` so "absent" means *fall back to the coarser bucket* rather
/// than *free* — see [`crate::ModelInfo::cost`] for the fallback chain.
///
/// Field names use the canonical "input/output" vocabulary; legacy
/// "prompt/completion/cache_hit" names are accepted via serde aliases so
/// existing configs and the generated `models/cloud.toml` continue to load.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PricingConfig {
    /// Cost per million uncached input tokens in USD.
    #[serde(alias = "prompt_cost_per_million")]
    pub input_cost_per_million: f64,
    /// Cost per million output tokens in USD.
    #[serde(alias = "completion_cost_per_million")]
    pub output_cost_per_million: f64,

    /// Cost per million cache-read input tokens in USD.
    /// `None` → falls back to `input_cost_per_million`.
    #[serde(
        alias = "cache_hit_cost_per_million",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_cost_per_million: Option<f64>,
    /// Cost per million cache-write input tokens in USD (Anthropic charges
    /// ~1.25× of base input for this). `None` → falls back to
    /// `input_cost_per_million`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_cost_per_million: Option<f64>,
    /// Cost per million reasoning output tokens in USD.
    /// `None` → falls back to `output_cost_per_million`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_cost_per_million: Option<f64>,
    /// Cost per million audio input tokens in USD.
    /// `None` → falls back to `input_cost_per_million`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_input_cost_per_million: Option<f64>,
    /// Cost per million audio output tokens in USD.
    /// `None` → falls back to `output_cost_per_million`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_output_cost_per_million: Option<f64>,

    /// Per-call cost in USD for upstream-side tools like web search. Keyed by
    /// tool name (must match the names crabllm reports in
    /// [`crate::Usage::server_tool_calls`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub server_tool_cost_per_call: BTreeMap<String, f64>,
}

use crate::PricingConfig;
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
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(
        default,
        deserialize_with = "lenient_pricing",
        skip_serializing_if = "Option::is_none"
    )]
    pub pricing: Option<PricingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
    /// Native dialects for this model — the endpoints that serve it without
    /// translation. Empty is omitted for backward compatibility.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dialects: Vec<Dialect>,
}

/// A third party's `pricing` is its own schema — OpenRouter quotes per-token
/// strings where this is per-million floats. Keep ours across a round trip,
/// and drop theirs rather than misread it as a number it isn't.
fn lenient_pricing<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<PricingConfig>, D::Error> {
    let value = serde_json::Value::deserialize(d)?;
    Ok(serde_json::from_value::<Option<PricingConfig>>(value)
        .ok()
        .flatten())
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

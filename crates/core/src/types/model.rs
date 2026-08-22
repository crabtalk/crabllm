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

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PricingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
    /// Native dialects for this model — the endpoints that serve it without
    /// translation. Empty is omitted for backward compatibility.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dialects: Vec<Dialect>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ModelList {
    pub object: String,
    pub data: Vec<Model>,
}

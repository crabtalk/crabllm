use crate::PricingConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-model metadata: context window and token pricing.
///
/// Every field is `Option` so partial overrides work — a config entry
/// that sets only `context_length` inherits pricing from the static
/// default via [`ModelInfo::merge`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Maximum context window in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Token pricing (input + output cost per million tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PricingConfig>,
}

impl ModelInfo {
    /// Fill `None` fields from `fallback`. Pricing is atomic — override
    /// both input and output cost, or inherit both.
    pub fn merge(&mut self, fallback: &ModelInfo) {
        if self.context_length.is_none() {
            self.context_length = fallback.context_length;
        }
        if self.pricing.is_none() {
            self.pricing = fallback.pricing;
        }
    }

    /// Compute cost in USD for the given token counts. Returns 0.0 if
    /// pricing is not set.
    pub fn cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let Some(ref p) = self.pricing else {
            return 0.0;
        };
        (prompt_tokens as f64 * p.prompt_cost_per_million
            + completion_tokens as f64 * p.completion_cost_per_million)
            / 1_000_000.0
    }
}

/// Resolve model info from config overrides only.
pub fn resolve_model_info(
    model: &str,
    config_overrides: &HashMap<String, ModelInfo>,
) -> Option<ModelInfo> {
    resolve_model_info_full(model, &HashMap::new(), config_overrides)
}

/// Two-layer resolve: admin override > config override.
/// Admin fields take priority; missing fields fall through to config.
pub fn resolve_model_info_full(
    model: &str,
    admin_overrides: &HashMap<String, ModelInfo>,
    config_overrides: &HashMap<String, ModelInfo>,
) -> Option<ModelInfo> {
    let admin = admin_overrides.get(model);
    let config = config_overrides.get(model);

    match (admin, config) {
        (None, None) => None,
        (Some(a), None) => Some(a.clone()),
        (None, Some(c)) => Some(c.clone()),
        (Some(a), Some(c)) => {
            let mut merged = a.clone();
            merged.merge(c);
            Some(merged)
        }
    }
}

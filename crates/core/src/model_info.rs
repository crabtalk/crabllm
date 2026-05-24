use crate::{PricingConfig, Usage};
use serde::{Deserialize, Serialize};

/// Per-model metadata: context window and token pricing.
///
/// Every field is `Option` so partial overrides work — a config entry
/// that sets only `context_length` can leave pricing unset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ModelInfo {
    /// Maximum context window in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Token pricing (per-axis costs per million tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PricingConfig>,
    /// Whether the model accepts image/video input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
}

impl ModelInfo {
    /// Compute cost in USD for the given usage. Returns 0.0 when pricing is
    /// unset.
    ///
    /// Each axis is priced independently. Secondary rates fall back to the
    /// next-coarser bucket when unset:
    /// - `cache_read`, `cache_write`, `audio_input` → `input_cost_per_million`
    /// - `reasoning`, `audio_output` → `output_cost_per_million`
    ///
    /// `None` never means "free" — that would silently zero out billing on any
    /// axis where pricing data is incomplete.
    pub fn cost(&self, u: &Usage) -> f64 {
        let Some(ref p) = self.pricing else {
            return 0.0;
        };
        let input_rate = p.input_cost_per_million;
        let output_rate = p.output_cost_per_million;
        let cache_read_rate = p.cache_read_cost_per_million.unwrap_or(input_rate);
        let cache_write_rate = p.cache_write_cost_per_million.unwrap_or(input_rate);
        let reasoning_rate = p.reasoning_cost_per_million.unwrap_or(output_rate);

        let tokens = u.input_tokens as f64 * input_rate
            + u.cache_read_tokens as f64 * cache_read_rate
            + u.cache_write_tokens as f64 * cache_write_rate
            + u.output_tokens as f64 * output_rate
            + u.reasoning_tokens as f64 * reasoning_rate;
        let mut total = tokens / 1_000_000.0;

        for (tool, calls) in &u.server_tool_calls {
            if let Some(rate) = p.server_tool_cost_per_call.get(tool) {
                total += *calls as f64 * rate;
            }
        }
        total
    }
}

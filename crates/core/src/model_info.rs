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

const fn info(context_length: u32, prompt: f64, completion: f64) -> ModelInfo {
    ModelInfo {
        context_length: Some(context_length),
        pricing: Some(PricingConfig {
            prompt_cost_per_million: prompt,
            completion_cost_per_million: completion,
        }),
    }
}

/// Look up static defaults for a well-known model.
pub fn default_model_info(model: &str) -> Option<ModelInfo> {
    Some(match model {
        // OpenAI — GPT-4o family
        "gpt-4o" | "gpt-4o-2024-11-20" => info(128_000, 2.50, 10.00),
        "gpt-4o-2024-08-06" => info(128_000, 2.50, 10.00),
        "gpt-4o-2024-05-13" => info(128_000, 5.00, 15.00),
        "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => info(128_000, 0.15, 0.60),
        "gpt-4o-audio-preview" => info(128_000, 2.50, 10.00),
        "gpt-4o-search-preview" => info(128_000, 2.50, 10.00),

        // OpenAI — GPT-4.1 family
        "gpt-4.1" | "gpt-4.1-2025-04-14" => info(1_047_576, 2.00, 8.00),
        "gpt-4.1-mini" | "gpt-4.1-mini-2025-04-14" => info(1_047_576, 0.40, 1.60),
        "gpt-4.1-nano" | "gpt-4.1-nano-2025-04-14" => info(1_047_576, 0.10, 0.40),

        // OpenAI — o-series reasoning
        "o1" | "o1-2024-12-17" => info(200_000, 15.00, 60.00),
        "o1-mini" | "o1-mini-2024-09-12" => info(128_000, 1.10, 4.40),
        "o1-pro" | "o1-pro-2025-03-19" => info(200_000, 150.00, 600.00),
        "o3" | "o3-2025-04-16" => info(200_000, 2.00, 8.00),
        "o3-mini" | "o3-mini-2025-01-31" => info(200_000, 1.10, 4.40),
        "o4-mini" | "o4-mini-2025-04-16" => info(200_000, 1.10, 4.40),

        // OpenAI — GPT-4 Turbo / legacy
        "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" => info(128_000, 10.00, 30.00),
        "gpt-4" | "gpt-4-0613" => info(8_192, 30.00, 60.00),
        "gpt-4-32k" => info(32_768, 60.00, 120.00),

        // OpenAI — GPT-3.5
        "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" => info(16_385, 0.50, 1.50),

        // OpenAI — embeddings
        "text-embedding-3-small" => info(8_191, 0.02, 0.0),
        "text-embedding-3-large" => info(8_191, 0.13, 0.0),
        "text-embedding-ada-002" => info(8_191, 0.10, 0.0),

        // Anthropic — Claude 4 family
        "claude-opus-4-20250514" => info(200_000, 15.00, 75.00),
        "claude-sonnet-4-20250514" => info(200_000, 3.00, 15.00),

        // Anthropic — Claude 3.7
        "claude-3-7-sonnet-20250219" | "claude-3-7-sonnet-latest" => info(200_000, 3.00, 15.00),

        // Anthropic — Claude 3.5
        "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet-latest" => info(200_000, 3.00, 15.00),
        "claude-3-5-sonnet-20240620" => info(200_000, 3.00, 15.00),
        "claude-3-5-haiku-20241022" | "claude-3-5-haiku-latest" => info(200_000, 0.80, 4.00),

        // Anthropic — Claude 3
        "claude-3-opus-20240229" | "claude-3-opus-latest" => info(200_000, 15.00, 75.00),
        "claude-3-haiku-20240307" => info(200_000, 0.25, 1.25),

        // Google — Gemini 2.5
        "gemini-2.5-pro" | "gemini-2.5-pro-preview-05-06" => info(1_048_576, 1.25, 10.00),
        "gemini-2.5-flash" | "gemini-2.5-flash-preview-04-17" => info(1_048_576, 0.15, 0.60),

        // Google — Gemini 2.0
        "gemini-2.0-flash" => info(1_048_576, 0.10, 0.40),
        "gemini-2.0-flash-lite" => info(1_048_576, 0.075, 0.30),

        // Google — Gemini 1.5
        "gemini-1.5-pro" => info(2_097_152, 1.25, 5.00),
        "gemini-1.5-flash" => info(1_048_576, 0.075, 0.30),

        // Mistral
        "mistral-large-latest" => info(128_000, 2.00, 6.00),
        "mistral-small-latest" => info(32_000, 0.10, 0.30),
        "codestral-latest" => info(256_000, 0.30, 0.90),
        "mistral-embed" => info(8_192, 0.10, 0.0),

        // Meta — Llama 4
        "llama-4-scout-17b-16e-instruct" => info(512_000, 0.15, 0.60),
        "llama-4-maverick-17b-128e-instruct" => info(256_000, 0.25, 1.00),

        // Meta — Llama 3.3 / 3.1
        "llama-3.3-70b-instruct" => info(128_000, 0.18, 0.36),
        "llama-3.1-405b-instruct" => info(128_000, 0.80, 0.80),
        "llama-3.1-70b-instruct" => info(128_000, 0.18, 0.18),
        "llama-3.1-8b-instruct" => info(128_000, 0.05, 0.08),

        // DeepSeek
        "deepseek-chat" => info(128_000, 0.14, 0.28),
        "deepseek-reasoner" => info(128_000, 0.55, 2.19),

        _ => return None,
    })
}

/// Resolve model info by merging config overrides with static defaults.
/// Config fields take priority; missing fields fall through to defaults.
pub fn resolve_model_info(
    model: &str,
    config_overrides: &HashMap<String, ModelInfo>,
) -> Option<ModelInfo> {
    let config = config_overrides.get(model);
    let default = default_model_info(model);

    match (config, default) {
        (Some(cfg), Some(def)) => {
            let mut merged = cfg.clone();
            merged.merge(&def);
            Some(merged)
        }
        (Some(cfg), None) => Some(cfg.clone()),
        (None, def) => def,
    }
}

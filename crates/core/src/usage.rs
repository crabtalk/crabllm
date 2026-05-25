use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Canonical token usage. Axes are disjoint — billing computes cost as a sum
/// over (axis, rate) pairs without any subtraction or clamping.
///
/// Input axes
/// - `input_tokens`: uncached new input (the portion the model has to read fresh)
/// - `cache_read_tokens`: served from prompt cache (cheaper than base input)
/// - `cache_write_tokens`: written to prompt cache this turn (often *more*
///   expensive than base input — Anthropic charges ~1.25× for cache writes)
///
/// Output axes
/// - `output_tokens`: regular completion tokens, excluding reasoning
/// - `reasoning_tokens`: thinking/reasoning tokens (may have a separate rate)
///
/// Side-channel
/// - `server_tool_calls`: per-call billing for upstream-side tools like web
///   search. Keyed by tool name (`"web_search"`, etc.).
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Usage {
    pub input_tokens: u32,
    #[serde(default)]
    pub cache_read_tokens: u32,
    #[serde(default)]
    pub cache_write_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub reasoning_tokens: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub server_tool_calls: BTreeMap<String, u32>,
}

impl Usage {
    /// Total prompt size (cached + uncached + writes). Equivalent to OpenAI's
    /// `prompt_tokens`.
    pub fn prompt_tokens(&self) -> u32 {
        self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    /// Total completion size including reasoning. Equivalent to OpenAI's
    /// `completion_tokens`.
    pub fn completion_tokens(&self) -> u32 {
        self.output_tokens + self.reasoning_tokens
    }

    /// Prompt plus completion — equivalent to OpenAI's `total_tokens`.
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens() + self.completion_tokens()
    }
}

mod peek {
    use crate::types::{AnthropicUsage, GeminiUsage, OpenAiUsage};
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct OpenAi {
        pub usage: Option<OpenAiUsage>,
    }

    #[derive(Deserialize)]
    pub struct Anthropic {
        pub usage: Option<AnthropicUsage>,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Gemini {
        pub usage_metadata: Option<GeminiUsage>,
    }
}

impl From<&[u8]> for Usage {
    fn from(raw: &[u8]) -> Self {
        if let Ok(peek::OpenAi { usage: Some(u) }) = crate::json::from_slice(raw) {
            if u.prompt_tokens > 0 || u.completion_tokens > 0 {
                return Usage::from(&u);
            }
        }

        if let Ok(peek::Anthropic { usage: Some(u) }) = crate::json::from_slice(raw) {
            if u.input_tokens > 0 || u.output_tokens > 0 {
                return Usage::from(&u);
            }
        }

        if let Ok(peek::Gemini {
            usage_metadata: Some(u),
        }) = crate::json::from_slice(raw)
        {
            if u.total_token_count > 0 {
                return Usage::from(&u);
            }
        }

        Usage::default()
    }
}

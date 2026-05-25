use crate::Usage;
use crate::types::gemini::GeminiContent;
use crate::types::openai::FinishReason;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    #[serde(default)]
    pub candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    pub usage_metadata: Option<GeminiUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GeminiFinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCandidate {
    #[serde(default)]
    pub content: Option<GeminiContent>,
    #[serde(default)]
    pub finish_reason: Option<GeminiFinishReason>,
}

/// Gemini wire-format usage. Field semantics follow Google's convention:
/// `prompt_token_count` is the *total* prompt size (includes cached portion);
/// `cached_content_token_count` is the cached subset. Convert to canonical
/// [`Usage`] for any internal billing or metering use.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsage {
    #[serde(default)]
    pub prompt_token_count: u32,
    #[serde(default)]
    pub candidates_token_count: u32,
    #[serde(default)]
    pub total_token_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
}

impl From<&GeminiFinishReason> for FinishReason {
    fn from(r: &GeminiFinishReason) -> Self {
        match r {
            GeminiFinishReason::Stop => FinishReason::Stop,
            GeminiFinishReason::MaxTokens => FinishReason::Length,
            GeminiFinishReason::Safety
            | GeminiFinishReason::Blocklist
            | GeminiFinishReason::ProhibitedContent
            | GeminiFinishReason::Spii => FinishReason::ContentFilter,
            GeminiFinishReason::Recitation => FinishReason::Custom("recitation".into()),
            GeminiFinishReason::MalformedFunctionCall => {
                FinishReason::Custom("malformed_function_call".into())
            }
            GeminiFinishReason::Other => FinishReason::Custom("other".into()),
        }
    }
}

impl From<&GeminiUsage> for Usage {
    fn from(u: &GeminiUsage) -> Self {
        // Gemini's `prompt_token_count` is the total prompt; the cached subset
        // is reported separately. Split into canonical input + cache_read.
        // Gemini does not expose cache-write counts.
        let cache_read = u.cached_content_token_count.unwrap_or(0);
        let input = u.prompt_token_count.saturating_sub(cache_read);
        let reasoning = u.thoughts_token_count.unwrap_or(0);
        Self {
            input_tokens: input,
            cache_read_tokens: cache_read,
            cache_write_tokens: 0,
            // Gemini's `candidates_token_count` is regular output only.
            // `thoughts_token_count` (when present) is the reasoning portion.
            output_tokens: u.candidates_token_count,
            reasoning_tokens: reasoning,
            server_tool_calls: Default::default(),
        }
    }
}

impl From<&Usage> for GeminiUsage {
    fn from(u: &Usage) -> Self {
        Self {
            prompt_token_count: u.prompt_tokens(),
            candidates_token_count: u.output_tokens,
            total_token_count: u.total_tokens(),
            cached_content_token_count: if u.cache_read_tokens > 0 {
                Some(u.cache_read_tokens)
            } else {
                None
            },
            thoughts_token_count: if u.reasoning_tokens > 0 {
                Some(u.reasoning_tokens)
            } else {
                None
            },
        }
    }
}

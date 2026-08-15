use crate::types::gemini::Content;
use crate::types::openai;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Response {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub usage_metadata: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
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
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// Gemini wire-format usage. Field semantics follow Google's convention:
/// `prompt_token_count` is the *total* prompt size (includes cached portion);
/// `cached_content_token_count` is the cached subset. Convert to canonical
/// [`Usage`] for any internal billing or metering use.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
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

impl From<&str> for FinishReason {
    fn from(reason: &str) -> Self {
        match reason {
            "end_turn" | "tool_use" | "stop_sequence" => Self::Stop,
            "max_tokens" => Self::MaxTokens,
            "content_filter" => Self::Safety,
            _ => Self::Other,
        }
    }
}

impl From<&openai::FinishReason> for FinishReason {
    fn from(r: &openai::FinishReason) -> Self {
        match r {
            openai::FinishReason::Stop | openai::FinishReason::ToolCalls => Self::Stop,
            openai::FinishReason::Length => Self::MaxTokens,
            openai::FinishReason::ContentFilter => Self::Safety,
            openai::FinishReason::Custom(_) => Self::Other,
        }
    }
}

impl From<&FinishReason> for openai::FinishReason {
    fn from(r: &FinishReason) -> Self {
        match r {
            FinishReason::Stop => Self::Stop,
            FinishReason::MaxTokens => Self::Length,
            FinishReason::Safety
            | FinishReason::Blocklist
            | FinishReason::ProhibitedContent
            | FinishReason::Spii => Self::ContentFilter,
            FinishReason::Recitation => Self::Custom("recitation".into()),
            FinishReason::MalformedFunctionCall => Self::Custom("malformed_function_call".into()),
            FinishReason::Other => Self::Custom("other".into()),
        }
    }
}

impl From<&Usage> for crate::Usage {
    fn from(u: &Usage) -> Self {
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

impl From<&crate::Usage> for Usage {
    fn from(u: &crate::Usage) -> Self {
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

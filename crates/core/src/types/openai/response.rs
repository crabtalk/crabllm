use crate::Usage;
use crate::types::openai::{FinishReason, Message, ToolType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

impl ChatCompletionResponse {
    /// First choice's message, if present.
    pub fn message(&self) -> Option<&Message> {
        self.choices.first().map(|c| &c.message)
    }

    /// Text content from the first choice's message, if non-empty.
    pub fn content(&self) -> Option<&str> {
        self.choices.first()?.message.content_str()
    }

    /// Reasoning content from the first choice's message, if non-empty.
    pub fn reasoning_content(&self) -> Option<&str> {
        self.choices.first()?.message.thinking()
    }

    /// Finish reason from the first choice, if present.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}

/// OpenAI wire-format usage. Field semantics follow OpenAI/DeepSeek-openai
/// convention: `prompt_tokens` is the total prompt including cache,
/// `prompt_cache_hit_tokens` is a subset. Convert to canonical [`Usage`] for
/// any internal billing or metering use.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
}

impl From<&Usage> for OpenAiUsage {
    fn from(u: &Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens(),
            completion_tokens: u.completion_tokens(),
            total_tokens: u.total_tokens(),
            completion_tokens_details: if u.reasoning_tokens > 0 {
                Some(CompletionTokensDetails {
                    reasoning_tokens: Some(u.reasoning_tokens),
                })
            } else {
                None
            },
            // Emitted in both shapes: OpenAI's own clients read
            // `prompt_tokens_details`, DeepSeek-shaped ones read the flat field.
            prompt_tokens_details: (u.cache_read_tokens > 0).then_some(PromptTokensDetails {
                cached_tokens: Some(u.cache_read_tokens),
            }),
            prompt_cache_hit_tokens: (u.cache_read_tokens > 0).then_some(u.cache_read_tokens),
            // `prompt_cache_miss_tokens` is semantically ambiguous on the
            // OpenAI wire (some providers mean "uncached input," some mean
            // "cache writes"). We never emit it — canonical [`Usage`] carries
            // the unambiguous components.
            prompt_cache_miss_tokens: None,
        }
    }
}

impl From<&OpenAiUsage> for Usage {
    fn from(u: &OpenAiUsage) -> Self {
        // `prompt_tokens` is the total prompt (cached + uncached); the cached
        // subset arrives either as OpenAI's `prompt_tokens_details.cached_tokens`
        // or DeepSeek's flat `prompt_cache_hit_tokens`. Cache *writes* are left
        // at zero: `prompt_tokens_details.cache_write_tokens` exists but its
        // relationship to `prompt_tokens` isn't documented clearly enough to
        // bill on. Reasoning lives in completion details.
        let cache_read = u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .or(u.prompt_cache_hit_tokens)
            .unwrap_or(0);
        let input = u.prompt_tokens.saturating_sub(cache_read);
        let reasoning = u
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .unwrap_or(0);
        let output = u.completion_tokens.saturating_sub(reasoning);
        Self {
            input_tokens: input,
            cache_read_tokens: cache_read,
            cache_write_tokens: 0,
            output_tokens: output,
            reasoning_tokens: reasoning,
            server_tool_calls: Default::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

/// Where OpenAI itself reports prompt cache hits. DeepSeek and the providers
/// that copied it use the flat `prompt_cache_hit_tokens` instead, so both have
/// to be read — missing this one meters every cache hit as full-price input.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionCall,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

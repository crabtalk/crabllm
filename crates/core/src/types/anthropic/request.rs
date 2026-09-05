use crate::types::anthropic::{ContentBlock, Message};
use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The `anthropic-version` header value the gateway and its clients send on
/// Anthropic Messages requests. Single source so the two can't drift.
pub const VERSION: &str = "2023-06-01";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: String,
    /// The pre-4.7 dialect's integer budget. Absent on `adaptive`, which
    /// carries its depth in [`OutputConfig::effort`] instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OutputConfig {
    /// `low` | `medium` | `high` | `xhigh` | `max`. Anthropic defaults to
    /// `high` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(as = AnthropicRequest))]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<System>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

/// System prompt: either a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(as = AnthropicSystem))]
#[serde(untagged)]
pub enum System {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(as = AnthropicTool))]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub input_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

use crate::types::anthropic::ContentBlock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Response {
    pub id: String,
    #[serde(default = "default_message_type")]
    pub r#type: String,
    #[serde(default = "default_assistant_role")]
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

fn default_message_type() -> String {
    "message".to_string()
}

fn default_assistant_role() -> String {
    "assistant".to_string()
}

/// Anthropic wire-format usage. Field semantics follow Anthropic convention:
/// `input_tokens` is *only* the uncached new portion; the three input fields
/// are additive components of the total prompt. Convert to canonical [`Usage`]
/// for any internal billing or metering use.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
}

impl From<&Usage> for crate::Usage {
    fn from(u: &Usage) -> Self {
        Self {
            input_tokens: u.input_tokens,
            cache_read_tokens: u.cache_read_input_tokens.unwrap_or(0),
            cache_write_tokens: u.cache_creation_input_tokens.unwrap_or(0),
            output_tokens: u.output_tokens,
            // Anthropic bills thinking as part of output_tokens — there is no
            // separate reasoning counter on the wire.
            reasoning_tokens: 0,
            server_tool_calls: Default::default(),
        }
    }
}

impl From<&crate::Usage> for Usage {
    fn from(u: &crate::Usage) -> Self {
        Self {
            input_tokens: u.input_tokens,
            // Anthropic's output_tokens includes thinking, so fold reasoning
            // back into the wire output count.
            output_tokens: u.output_tokens + u.reasoning_tokens,
            cache_read_input_tokens: if u.cache_read_tokens > 0 {
                Some(u.cache_read_tokens)
            } else {
                None
            },
            cache_creation_input_tokens: if u.cache_write_tokens > 0 {
                Some(u.cache_write_tokens)
            } else {
                None
            },
        }
    }
}

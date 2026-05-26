pub use request::{ChatCompletionRequest, FunctionDef, Stop, Tool, ToolChoice};
pub use response::{
    ChatCompletionResponse, Choice, CompletionTokensDetails, FunctionCall, OpenAiUsage, ToolCall,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub use stream::{ChatCompletionChunk, ChunkChoice, Delta, FunctionCallDelta, ToolCallDelta};

mod request;
mod response;
mod stream;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
    Developer,
    Custom(String),
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Tool => "tool",
            Self::Developer => "developer",
            Self::Custom(s) => s,
        }
    }
}

impl Serialize for Role {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "tool" => Self::Tool,
            "developer" => Self::Developer,
            _ => Self::Custom(s),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Custom(String),
}

impl FinishReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Custom(s) => s,
        }
    }

    pub fn from_anthropic_stop(reason: &str) -> Self {
        match reason {
            "end_turn" => Self::Stop,
            "max_tokens" => Self::Length,
            "tool_use" => Self::ToolCalls,
            other => Self::Custom(other.to_string()),
        }
    }

    pub fn to_anthropic_stop(&self) -> String {
        match self {
            Self::Stop => "end_turn".to_string(),
            Self::Length => "max_tokens".to_string(),
            Self::ToolCalls => "tool_use".to_string(),
            Self::ContentFilter => "content_filter".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }
}

impl Serialize for FinishReason {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            _ => Self::Custom(s),
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum ToolType {
    #[serde(rename = "function")]
    #[default]
    Function,
}

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "image")]
    Image {
        source: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text {
            text: s.into(),
            cache_control: None,
        }
    }
}

/// Tool result content: either a plain string or nested content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "openapi", schema(no_recursion))]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(content)],
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(content)],
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::text(content)],
        }
    }

    pub fn tool(
        tool_use_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                name: Some(name.into()),
                content: ToolResultContent::Text(content.into()),
                cache_control: None,
            }],
        }
    }

    /// First non-empty text found in content blocks.
    ///
    /// Checks `Text` blocks first, then falls back to
    /// `ToolResult { content: Text(..) }` blocks.
    pub fn content_str(&self) -> Option<&str> {
        for block in &self.content {
            if let ContentBlock::Text { text, .. } = block
                && !text.is_empty()
            {
                return Some(text.as_str());
            }
        }
        for block in &self.content {
            if let ContentBlock::ToolResult {
                content: ToolResultContent::Text(s),
                ..
            } = block
                && !s.is_empty()
            {
                return Some(s.as_str());
            }
        }
        None
    }

    /// Iterator over tool-use blocks.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &serde_json::Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }

    /// The thinking content, if any.
    pub fn thinking(&self) -> Option<&str> {
        for block in &self.content {
            if let ContentBlock::Thinking { thinking, .. } = block
                && !thinking.is_empty()
            {
                return Some(thinking.as_str());
            }
        }
        None
    }
}

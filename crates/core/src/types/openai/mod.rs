pub use request::{ChatCompletionRequest, FunctionDef, Stop, StreamOptions, Tool, ToolChoice};
pub use response::{
    ChatCompletionResponse, Choice, CompletionTokensDetails, FunctionCall, OpenAiUsage,
    PromptTokensDetails, ToolCall,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub use stream::{ChatCompletionChunk, ChunkChoice, Delta, FunctionCallDelta, ToolCallDelta};

mod ir;
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

/// A chat message on the OpenAI wire. One struct covers both directions —
/// requests set `tool_call_id`/`name`, responses set `tool_calls` — which is
/// how OpenAI itself models it. `extra` preserves fields we don't model so a
/// provider extension survives a round trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Not an OpenAI field — the LiteLLM extension for aiming prompt caching
    /// at an Anthropic backend through an OpenAI-shaped request. Read on the
    /// way in, never sent onward, since OpenAI rejects unknown message keys.
    #[serde(default, skip_serializing)]
    pub cache_control: Option<serde_json::Value>,
    #[serde(flatten, default)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: Role::User,
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            cache_control: None,
            extra: serde_json::Map::new(),
        }
    }
}

impl Message {
    fn with_role(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(MessageContent::Text(content.into())),
            ..Default::default()
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::with_role(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::with_role(Role::Assistant, content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::with_role(Role::System, content)
    }

    /// A tool result. OpenAI carries these as their own `role: "tool"`
    /// message keyed by `tool_call_id`, not as a block inside a user message.
    /// A tool message takes exactly `role`, `content` and `tool_call_id` — it
    /// has no `name`, unlike the other roles.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        let mut msg = Self::with_role(Role::Tool, content);
        msg.tool_call_id = Some(tool_call_id.into());
        msg
    }

    /// Text content, or `None` when empty or absent. Multi-part content
    /// yields the first non-empty text part.
    pub fn content_str(&self) -> Option<&str> {
        match self.content.as_ref()? {
            MessageContent::Text(s) => (!s.is_empty()).then_some(s.as_str()),
            MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                ContentPart::Text { text } if !text.is_empty() => Some(text.as_str()),
                _ => None,
            }),
        }
    }

    /// Iterator over `(id, name, arguments)` of this message's tool calls.
    /// `arguments` is the raw JSON string OpenAI sends, not a parsed value.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &str)> {
        self.tool_calls.iter().flatten().map(|tc| {
            (
                tc.id.as_str(),
                tc.function.name.as_str(),
                tc.function.arguments.as_str(),
            )
        })
    }

    /// The reasoning content, if any. Named `reasoning_content` on the wire
    /// by DeepSeek and the providers that follow it.
    pub fn thinking(&self) -> Option<&str> {
        self.reasoning_content.as_deref().filter(|s| !s.is_empty())
    }
}

/// Message content: a plain string, or an array of typed parts for
/// multimodal input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A content part. `text` and `image_url` are what a user message may carry;
/// `refusal` only ever arrives on an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

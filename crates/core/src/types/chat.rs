use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ── Enums ──

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
pub enum ToolType {
    #[serde(rename = "function")]
    #[default]
    Function,
}

/// Controls which tool the model should call.
///
/// String variants (`Disabled`, `Auto`, `Required`) serialize as `"none"`,
/// `"auto"`, `"required"`. `Function` serializes as
/// `{"type":"function","function":{"name":"..."}}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "serde_json::Value", try_from = "serde_json::Value")]
pub enum ToolChoice {
    Disabled,
    #[default]
    Auto,
    Required,
    Function {
        name: String,
    },
}

impl From<ToolChoice> for serde_json::Value {
    fn from(tc: ToolChoice) -> Self {
        match tc {
            ToolChoice::Disabled => serde_json::Value::String("none".into()),
            ToolChoice::Auto => serde_json::Value::String("auto".into()),
            ToolChoice::Required => serde_json::Value::String("required".into()),
            ToolChoice::Function { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        }
    }
}

impl TryFrom<serde_json::Value> for ToolChoice {
    type Error = String;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match &value {
            serde_json::Value::String(s) => Ok(Self::from(s.as_str())),
            serde_json::Value::Object(map) => {
                let name = map
                    .get("function")
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                    .ok_or("missing function.name")?;
                Ok(Self::Function {
                    name: name.to_string(),
                })
            }
            _ => Err("expected string or object".into()),
        }
    }
}

impl From<&str> for ToolChoice {
    fn from(value: &str) -> Self {
        match value {
            "none" => Self::Disabled,
            "auto" => Self::Auto,
            "required" => Self::Required,
            name => Self::Function {
                name: name.to_string(),
            },
        }
    }
}

// ── Request ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(flatten, default)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(flatten, default)]
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Message {
    fn with_role(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(serde_json::Value::String(content.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Build a `Message` with role `User` and the given text content.
    pub fn user(content: impl Into<String>) -> Self {
        Self::with_role(Role::User, content)
    }

    /// Build a `Message` with role `Assistant` and the given text content.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::with_role(Role::Assistant, content)
    }

    /// Build a `Message` with role `System` and the given text content.
    pub fn system(content: impl Into<String>) -> Self {
        Self::with_role(Role::System, content)
    }

    /// Build a `Message` with role `Tool`, setting `tool_call_id` and `name`.
    pub fn tool(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let mut msg = Self::with_role(Role::Tool, content);
        msg.tool_call_id = Some(tool_call_id.into());
        msg.name = Some(name.into());
        msg
    }

    /// Return the text view of `content` when it is a non-empty JSON string.
    ///
    /// Returns `None` for `null`, empty strings, and non-string variants
    /// (e.g. multimodal arrays).
    pub fn content_str(&self) -> Option<&str> {
        self.content
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionCall,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionDef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

// ── Response ──

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

impl ChatCompletionResponse {
    /// Text content from the first choice's message, if non-empty.
    ///
    /// Empty strings collapse to `None` (via `Message::content_str`).
    pub fn content(&self) -> Option<&str> {
        self.choices.first()?.message.content_str()
    }

    /// Reasoning content from the first choice's message, if non-empty.
    ///
    /// Empty strings collapse to `None`.
    pub fn reasoning_content(&self) -> Option<&str> {
        self.choices
            .first()?
            .message
            .reasoning_content
            .as_deref()
            .filter(|s| !s.is_empty())
    }

    /// Tool calls from the first choice's message. Empty slice if none.
    pub fn tool_calls(&self) -> &[ToolCall] {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_deref())
            .unwrap_or(&[])
    }

    /// Finish reason from the first choice, if present.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

// ── Streaming ──

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

impl ChatCompletionChunk {
    /// Text delta from the first choice, if non-empty.
    ///
    /// Empty-string deltas (keepalives, boundary markers) collapse to `None`.
    pub fn content(&self) -> Option<&str> {
        self.choices
            .first()?
            .delta
            .content
            .as_deref()
            .filter(|s| !s.is_empty())
    }

    /// Reasoning-content delta from the first choice, if non-empty.
    ///
    /// Empty-string deltas collapse to `None`.
    pub fn reasoning_content(&self) -> Option<&str> {
        self.choices
            .first()?
            .delta
            .reasoning_content
            .as_deref()
            .filter(|s| !s.is_empty())
    }

    /// Tool-call deltas from the first choice. Empty slice if none.
    pub fn tool_calls(&self) -> &[ToolCallDelta] {
        self.choices
            .first()
            .and_then(|c| c.delta.tool_calls.as_deref())
            .unwrap_or(&[])
    }

    /// Finish reason from the first choice, if present.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub kind: Option<ToolType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

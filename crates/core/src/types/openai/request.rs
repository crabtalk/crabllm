use crate::types::openai::{Message, ToolType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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
    #[serde(skip)]
    pub thinking: Option<crate::types::anthropic::ThinkingConfig>,
    #[serde(skip)]
    pub anthropic_max_tokens: Option<u32>,
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionDef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

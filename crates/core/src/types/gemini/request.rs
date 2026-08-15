use crate::types::gemini::Content;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDef {
    pub function_declarations: Vec<FunctionDecl>,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionDecl {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Token budget for thinking. `0` disables it; `-1` lets the model decide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
    /// Whether thought summaries come back in the response parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

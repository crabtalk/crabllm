//! Google Gemini wire types (Generative Language API).
//!
//! Each SSE chunk from `:streamGenerateContent?alt=sse` is a full
//! [`GeminiResponse`] with a single candidate, so there are no separate
//! streaming wire types — the same response shape carries one chunk at a time.

pub use request::{GeminiFunctionDecl, GeminiRequest, GeminiToolDef, GenerationConfig};
pub use response::{GeminiCandidate, GeminiFinishReason, GeminiResponse, GeminiUsage};
use serde::{Deserialize, Serialize};

mod request;
mod response;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GeminiRole {
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<GeminiRole>,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,
    /// Gemini 2.5+ thinking-model marker for `functionCall` parts —
    /// must be echoed back unchanged on follow-up turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

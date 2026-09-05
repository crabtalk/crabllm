//! Google Gemini wire types (Generative Language API).
//!
//! Each SSE chunk from `:streamGenerateContent?alt=sse` is a full
//! [`Response`] with a single candidate, so there are no separate
//! streaming wire types — the same response shape carries one chunk at a time.

use alloc::{string::String, vec::Vec};
pub use request::{FunctionDecl, GenerationConfig, Request, ThinkingConfig, ToolDef};
pub use response::{Candidate, FinishReason, Response, Usage};
use serde::{Deserialize, Serialize};

mod ir;
mod request;
mod response;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// Gemini 2.5+ thinking-model marker for `functionCall` parts —
    /// must be echoed back unchanged on follow-up turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

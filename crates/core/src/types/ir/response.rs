use crate::Usage;
use crate::ir::Content;

#[derive(Debug, Clone)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub content: Vec<Content>,
    pub stop_reason: Option<StopReason>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    End,
    MaxTokens,
    ToolUse,
}

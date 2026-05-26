use crate::ir::{Content, Message};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub system: Option<Vec<Content>>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<Thinking>,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    Disabled,
    Named(String),
}

#[derive(Debug, Clone)]
pub struct Thinking {
    pub budget_tokens: Option<u32>,
}

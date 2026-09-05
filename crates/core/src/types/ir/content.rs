use alloc::{string::String, vec, vec::Vec};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub enum Content {
    Text(String),
    Image {
        media_type: String,
        data: String,
    },
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        call_id: String,
        content: Vec<Content>,
    },
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    /// A prompt-cache boundary: everything before this point is stable and
    /// worth caching.
    ///
    /// Anthropic spells this as `cache_control` on the block the prefix ends
    /// at, which reads like a property of that block but is really a mark
    /// between blocks — so the IR carries it as one. Formats without explicit
    /// caching drop it.
    CacheBreakpoint,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text(text.into())],
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.into())],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.into())],
        }
    }

    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(|c| match c {
            Content::Text(s) if !s.is_empty() => Some(s.as_str()),
            _ => None,
        })
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = (&str, &str, &Value)> {
        self.content.iter().filter_map(|c| match c {
            Content::ToolCall { id, name, input } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }

    pub fn reasoning(&self) -> Option<&str> {
        self.content.iter().find_map(|c| match c {
            Content::Reasoning { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
    }
}

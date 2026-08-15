use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Message {
    pub role: String,
    pub content: Content,
}

impl Message {
    /// Borrow the block list, or an empty slice if content is plain text.
    pub(crate) fn blocks(&self) -> &[ContentBlock] {
        match &self.content {
            Content::Blocks(b) => b,
            Content::Text(_) => &[],
        }
    }

    /// Mutably borrow the block list, or `None` if content is plain text.
    pub(crate) fn blocks_mut(&mut self) -> Option<&mut Vec<ContentBlock>> {
        match &mut self.content {
            Content::Blocks(b) => Some(b),
            Content::Text(_) => None,
        }
    }

    /// Iterator over `(id, name)` of `tool_use` blocks in this assistant
    /// message. Empty for non-assistant messages and for messages whose
    /// content is plain text.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        let is_assistant = self.role == "assistant";
        self.blocks().iter().filter_map(move |b| match b {
            ContentBlock::ToolUse { id, name, .. } if is_assistant => {
                Some((id.as_str(), name.as_str()))
            }
            _ => None,
        })
    }

    /// Iterator over `tool_use_id` of `tool_result` blocks in this user
    /// message. Empty for non-user messages and for plain-text content.
    pub fn tool_result_ids(&self) -> impl Iterator<Item = &str> + '_ {
        let is_user = self.role == "user";
        self.blocks().iter().filter_map(move |b| match b {
            ContentBlock::ToolResult { tool_use_id, .. } if is_user => Some(tool_use_id.as_str()),
            _ => None,
        })
    }

    /// True when this is a user message whose blocks are all `tool_result`s.
    /// Used to recognise dispatch-result messages that can be coalesced.
    pub fn is_tool_result_only_user(&self) -> bool {
        if self.role != "user" {
            return false;
        }
        let blocks = self.blocks();
        !blocks.is_empty()
            && blocks
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. }))
    }
}

/// Message content: either a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
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

    /// Whether this block ends a cacheable prefix. `thinking` blocks have no
    /// slot for it — Anthropic never caches to one.
    pub fn is_cache_breakpoint(&self) -> bool {
        match self {
            Self::Text { cache_control, .. }
            | Self::ToolUse { cache_control, .. }
            | Self::ToolResult { cache_control, .. }
            | Self::Image { cache_control, .. } => cache_control.is_some(),
            Self::Thinking { .. } => false,
        }
    }

    /// Mark this block as the end of a cacheable prefix.
    pub fn mark_cache_breakpoint(&mut self) {
        let marker = serde_json::json!({"type": "ephemeral"});
        match self {
            Self::Text { cache_control, .. }
            | Self::ToolUse { cache_control, .. }
            | Self::ToolResult { cache_control, .. }
            | Self::Image { cache_control, .. } => *cache_control = Some(marker),
            Self::Thinking { .. } => {}
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

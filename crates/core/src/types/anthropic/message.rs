use crate::types::openai::ContentBlock;
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

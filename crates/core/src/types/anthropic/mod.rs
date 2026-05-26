//! Anthropic Messages API wire types.
//!
//! The canonical content-block types (`ContentBlock`, `ToolResultContent`)
//! live in `crate::types::openai`. This module re-exports `ContentBlock` as
//! `AnthropicContentBlock` for backward compatibility and defines the
//! Anthropic-specific request/response envelope types.

pub use message::{AnthropicContent, AnthropicMessage};
pub use messages::AnthropicMessages;
pub use request::{
    AnthropicRequest, AnthropicSystem, AnthropicTool, DEFAULT_MAX_TOKENS, ThinkingConfig,
};
pub use response::{AnthropicResponse, AnthropicUsage};
pub use stream::{AnthropicStreamEvent, BlockDelta, MessageDeltaPayload};

use crate::types::openai::ContentBlock;

mod ir;
mod message;
mod messages;
mod request;
mod response;
mod stream;

pub type AnthropicContentBlock = ContentBlock;

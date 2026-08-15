//! Anthropic Messages API wire types.
//!
//! The canonical content-block types (`ContentBlock`, `ToolResultContent`)
//! live in `crate::types::openai`. This module re-exports `ContentBlock` as
//! `ContentBlock` for backward compatibility and defines the
//! Anthropic-specific request/response envelope types.

pub use convert::tool_choice;
pub use message::{Content, Message};
pub use messages::Messages;
pub use request::{DEFAULT_MAX_TOKENS, Request, System, ThinkingConfig, Tool, VERSION};
pub use response::{Response, Usage};
pub use stream::{BlockDelta, MessageDeltaPayload, StreamEvent};

mod convert;
mod ir;
mod message;
mod messages;
mod request;
mod response;
mod stream;

pub type ContentBlock = crate::types::openai::ContentBlock;

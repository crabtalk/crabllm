//! Anthropic-compatible inbound endpoint.
//!
//! When a client sends a request in Anthropic Messages API format, these
//! modules normalize it to the internal OpenAI-shaped `ChatCompletionRequest`
//! so the rest of the proxy pipeline (extensions, provider dispatch) is
//! unchanged, then translate the response back to Anthropic's wire format.

pub use handler::messages;
pub use sse::{AnthropicSseEvent, to_anthropic_sse};
pub use translate::{from_chat_completion, to_chat_completion};

mod handler;
mod sse;
mod translate;

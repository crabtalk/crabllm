//! Anthropic-compatible inbound endpoint.
//!
//! When a client sends a request in Anthropic Messages API format, these
//! modules normalize it to the internal OpenAI-shaped `ChatCompletionRequest`
//! so the rest of the proxy pipeline (extensions, provider dispatch) is
//! unchanged, then translate the response back to Anthropic's wire format.

pub use translate::to_chat_completion;

mod translate;

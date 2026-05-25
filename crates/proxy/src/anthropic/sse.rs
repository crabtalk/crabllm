//! Outbound SSE event model for Anthropic Messages streaming responses, plus
//! a stream adapter that folds an OpenAI-shaped chunk stream into this model.
//!
//! Delegates to [`crabllm_provider::chunks_to_anthropic_events`] for the
//! actual chunk-to-event conversion. This module re-exports the core
//! [`AnthropicStreamEvent`] type as `AnthropicSseEvent` for backward
//! compatibility with existing handler code.

pub use crabllm_core::AnthropicStreamEvent as AnthropicSseEvent;

use crabllm_core::{ChatCompletionChunk, Error};
use crabllm_provider::chunks_to_anthropic_events;
use futures::Stream;

/// Fold an internal chunk stream into Anthropic SSE events.
pub fn to_anthropic_sse(
    chunks: impl Stream<Item = Result<ChatCompletionChunk, Error>> + Unpin + Send + 'static,
) -> impl Stream<Item = Result<AnthropicSseEvent, Error>> + Send + 'static {
    chunks_to_anthropic_events(chunks)
}

//! Outbound SSE event model for Anthropic Messages streaming responses, plus
//! a stream adapter that folds an OpenAI-shaped chunk stream into this model.
//!
//! Anthropic's `/v1/messages` streaming format is structured around *content
//! blocks* (text, thinking, tool_use) with explicit open/close events. OpenAI's
//! chunk stream is flat — deltas mention their kind but never announce block
//! boundaries. The adapter below reconstructs block boundaries by watching
//! which delta kind the current chunk carries and closing the previous block
//! whenever the kind changes.
//!
//! # Usage accounting
//!
//! Anthropic's spec puts input_tokens on `message_start.usage` and output_tokens
//! on `message_delta.usage`. We don't know the provider's token accounting
//! until the *last* chunk arrives, so `message_start.usage` is emitted with
//! all-zero counts and the final usage (including input_tokens) is delivered
//! on `message_delta.usage`. Both the Python and TypeScript Anthropic SDKs
//! sum these two into a final `Usage`, so the totals are correct.

use crabllm_core::{
    AnthropicContentBlock, AnthropicResponse, AnthropicUsage, ChatCompletionChunk, Error,
    OpenAiUsage,
};
use futures::{Stream, StreamExt, stream};
use serde::Serialize;
use std::collections::VecDeque;

// ── Emission event types ──

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicSseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: BlockDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaPayload,
        usage: AnthropicUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
}

impl AnthropicSseEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop => "message_stop",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum BlockDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

// ── Adapter ──

/// Fold an internal chunk stream into Anthropic SSE events.
pub fn to_anthropic_sse(
    chunks: impl Stream<Item = Result<ChatCompletionChunk, Error>> + Unpin + Send + 'static,
) -> impl Stream<Item = Result<AnthropicSseEvent, Error>> + Send + 'static {
    let state = AdapterState::default();
    stream::unfold((chunks, state), |(mut chunks, mut state)| async move {
        loop {
            if let Some(event) = state.pending.pop_front() {
                return Some((Ok(event), (chunks, state)));
            }
            if let Some(err) = state.deferred_error.take() {
                state.finished = true;
                return Some((Err(err), (chunks, state)));
            }
            if state.finished {
                return None;
            }
            match chunks.next().await {
                Some(Ok(chunk)) => state.handle_chunk(chunk),
                Some(Err(e)) => state.abort_with_error(e),
                None => state.finalize("end_turn".to_string()),
            }
        }
    })
}

#[derive(Default)]
struct AdapterState {
    started: bool,
    finished: bool,
    current: Option<CurrentBlock>,
    next_index: u32,
    pending: VecDeque<AnthropicSseEvent>,
    deferred_error: Option<Error>,
    latest_usage: Option<OpenAiUsage>,
    stop_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentBlock {
    Text { index: u32 },
    Thinking { index: u32 },
    ToolUse { index: u32, openai_index: u32 },
}

impl AdapterState {
    fn handle_chunk(&mut self, chunk: ChatCompletionChunk) {
        self.ensure_started(&chunk);

        if let Some(usage) = chunk.usage {
            self.latest_usage = Some(usage);
        }

        let Some(choice) = chunk.choices.into_iter().next() else {
            return;
        };
        let delta = choice.delta;

        if let Some(reasoning) = delta.reasoning_content
            && !reasoning.is_empty()
        {
            let index = self.switch_to_thinking();
            self.pending
                .push_back(AnthropicSseEvent::ContentBlockDelta {
                    index,
                    delta: BlockDelta::Thinking {
                        thinking: reasoning,
                    },
                });
        }

        if let Some(text) = delta.content
            && !text.is_empty()
        {
            let index = self.switch_to_text();
            self.pending
                .push_back(AnthropicSseEvent::ContentBlockDelta {
                    index,
                    delta: BlockDelta::Text { text },
                });
        }

        if let Some(tool_deltas) = delta.tool_calls {
            for tc in tool_deltas {
                self.handle_tool_delta(tc);
            }
        }

        if let Some(reason) = choice.finish_reason {
            self.stop_reason = Some(finish_reason_to_stop(&reason));
        }
    }

    fn ensure_started(&mut self, chunk: &ChatCompletionChunk) {
        if self.started {
            return;
        }
        self.started = true;
        let msg = AnthropicResponse {
            id: chunk.id.clone(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            model: chunk.model.clone(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: empty_usage(),
        };
        self.pending
            .push_back(AnthropicSseEvent::MessageStart { message: msg });
    }

    fn close_current(&mut self) {
        if let Some(block) = self.current.take() {
            let index = match block {
                CurrentBlock::Text { index }
                | CurrentBlock::Thinking { index }
                | CurrentBlock::ToolUse { index, .. } => index,
            };
            self.pending
                .push_back(AnthropicSseEvent::ContentBlockStop { index });
        }
    }

    fn switch_to_text(&mut self) -> u32 {
        if let Some(CurrentBlock::Text { index }) = self.current {
            return index;
        }
        self.close_current();
        let index = self.allocate_index();
        self.pending
            .push_back(AnthropicSseEvent::ContentBlockStart {
                index,
                content_block: AnthropicContentBlock::text(""),
            });
        self.current = Some(CurrentBlock::Text { index });
        index
    }

    fn switch_to_thinking(&mut self) -> u32 {
        if let Some(CurrentBlock::Thinking { index }) = self.current {
            return index;
        }
        self.close_current();
        let index = self.allocate_index();
        self.pending
            .push_back(AnthropicSseEvent::ContentBlockStart {
                index,
                content_block: AnthropicContentBlock::Thinking {
                    thinking: String::new(),
                    signature: None,
                },
            });
        self.current = Some(CurrentBlock::Thinking { index });
        index
    }

    /// Open a fresh tool_use block. Always allocates a new Anthropic block
    /// index — we never reopen a closed block. Providers that stream each
    /// tool call contiguously (the common case) get the expected shape; a
    /// provider that interleaved deltas across tool indices would produce a
    /// valid but weird stream where the same tool spans multiple blocks.
    fn open_tool_use(&mut self, openai_index: u32, id: String, name: String) -> u32 {
        self.close_current();
        let index = self.allocate_index();
        self.pending
            .push_back(AnthropicSseEvent::ContentBlockStart {
                index,
                content_block: AnthropicContentBlock::ToolUse {
                    id,
                    name,
                    input: serde_json::json!({}),
                    cache_control: None,
                },
            });
        self.current = Some(CurrentBlock::ToolUse {
            index,
            openai_index,
        });
        index
    }

    fn handle_tool_delta(&mut self, tc: crabllm_core::ToolCallDelta) {
        let openai_index = tc.index;
        let current_index = match self.current {
            Some(CurrentBlock::ToolUse {
                index,
                openai_index: oi,
            }) if oi == openai_index => index,
            _ => {
                let id = tc.id.clone().unwrap_or_default();
                let name = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();
                self.open_tool_use(openai_index, id, name)
            }
        };

        if let Some(func) = tc.function
            && let Some(args) = func.arguments
            && !args.is_empty()
        {
            self.pending
                .push_back(AnthropicSseEvent::ContentBlockDelta {
                    index: current_index,
                    delta: BlockDelta::InputJson { partial_json: args },
                });
        }
    }

    fn allocate_index(&mut self) -> u32 {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }

    /// Normal end-of-stream: close current block, emit message_delta/stop.
    fn finalize(&mut self, default_stop: String) {
        debug_assert!(
            self.started,
            "finalize called before any chunk arrived — caller should have produced at least one chunk"
        );
        self.close_current();
        let stop_reason = self.stop_reason.take().or(Some(default_stop));
        let usage = self
            .latest_usage
            .take()
            .map(usage_to_anthropic)
            .unwrap_or_else(empty_usage);
        self.pending.push_back(AnthropicSseEvent::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason,
                stop_sequence: None,
            },
            usage,
        });
        self.pending.push_back(AnthropicSseEvent::MessageStop);
        self.finished = true;
    }

    /// Upstream error: flush a well-formed terminator *before* the error so
    /// SDK clients see message_stop and release their connection instead of
    /// hanging on a silently-aborted stream.
    fn abort_with_error(&mut self, err: Error) {
        if self.started {
            self.finalize("error".to_string());
        } else {
            // Haven't sent message_start yet — nothing to terminate. The
            // handler will convert the deferred error to an HTTP error.
            self.finished = true;
        }
        self.deferred_error = Some(err);
    }
}

fn empty_usage() -> AnthropicUsage {
    AnthropicUsage {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_input_tokens: None,
        cache_creation_input_tokens: None,
    }
}

fn usage_to_anthropic(u: OpenAiUsage) -> AnthropicUsage {
    AnthropicUsage::from(&crabllm_core::Usage::from(&u))
}

fn finish_reason_to_stop(reason: &crabllm_core::FinishReason) -> String {
    use crabllm_core::FinishReason::*;
    match reason {
        Stop => "end_turn".to_string(),
        Length => "max_tokens".to_string(),
        ToolCalls => "tool_use".to_string(),
        ContentFilter => "content_filter".to_string(),
        Custom(s) => s.clone(),
    }
}

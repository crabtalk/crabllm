use crate::{
    ByteStream, ChatCompletionChunk, ChunkChoice, Delta, Error, FinishReason, FunctionCallDelta,
    OpenAiUsage, Role, ToolCallDelta, ToolType, Usage, anthropic,
};
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use futures_util::stream::{self, Stream, StreamExt};
use serde::Deserialize;

// ── Anthropic SSE event types (parse-only) ──

#[derive(Deserialize)]
struct SseEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Option<SseDelta>,
    #[serde(default)]
    content_block: Option<SseContentBlock>,
    #[serde(default)]
    usage: Option<anthropic::Usage>,
    #[serde(default)]
    message: Option<SseMessage>,
    #[serde(default)]
    error: Option<SseError>,
}

#[derive(Deserialize)]
struct SseMessage {
    #[serde(default)]
    usage: Option<anthropic::Usage>,
}

#[derive(Deserialize)]
struct SseError {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
struct SseDelta {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
}

#[derive(Deserialize)]
struct SseContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

/// Map an Anthropic error `type` (from a streamed `error` event) to an HTTP
/// status, so the gateway's retry and client-facing status logic can treat a
/// mid-stream failure exactly like a non-stream one. `overloaded_error` maps to
/// 503 (not Anthropic's native 529) so it lands in the transient retry set.
fn anthropic_error_status(kind: &str) -> u16 {
    match kind {
        "invalid_request_error" => 400,
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "request_too_large" => 413,
        "rate_limit_error" => 429,
        "overloaded_error" => 503,
        _ => 500,
    }
}

/// Parse an Anthropic SSE byte stream into native `anthropic::StreamEvent`s.
pub fn anthropic_event_stream(
    byte_stream: ByteStream,
    model: String,
) -> impl Stream<Item = Result<anthropic::StreamEvent, Error>> {
    struct State {
        next_index: u32,
        input_usage: anthropic::Usage,
    }

    let lines = crate::codec::sse::data_lines(byte_stream).boxed();
    stream::unfold(
        (
            lines,
            model,
            State {
                next_index: 0,
                input_usage: anthropic::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            },
        ),
        |(mut lines, model, mut state)| async move {
            loop {
                let data = match lines.next().await {
                    Some(Ok(d)) => d,
                    Some(Err(e)) => return Some((Err(e), (lines, model, state))),
                    None => return None,
                };
                let Ok(event) = crate::json::from_str::<SseEvent>(&data) else {
                    continue;
                };

                match event.kind.as_str() {
                    "message_start" => {
                        if let Some(msg) = &event.message
                            && let Some(usage) = &msg.usage
                        {
                            state.input_usage = usage.clone();
                        }
                        let out = anthropic::StreamEvent::MessageStart {
                            message: anthropic::Response {
                                id: String::new(),
                                r#type: "message".to_string(),
                                role: "assistant".to_string(),
                                model: model.clone(),
                                content: Vec::new(),
                                stop_reason: None,
                                stop_sequence: None,
                                usage: state.input_usage.clone(),
                            },
                        };
                        return Some((Ok(out), (lines, model, state)));
                    }
                    "error" => {
                        let (status, body) = match &event.error {
                            Some(err) => (anthropic_error_status(&err.kind), err.message.clone()),
                            None => (502, "unknown stream error".to_string()),
                        };
                        return Some((
                            Err(Error::Provider {
                                status,
                                body,
                                retry_after: None,
                            }),
                            (lines, model, state),
                        ));
                    }
                    "content_block_start" => {
                        let Some(cb) = &event.content_block else {
                            continue;
                        };
                        let index = state.next_index;
                        state.next_index += 1;
                        let content_block = match cb.kind.as_str() {
                            "text" => anthropic::ContentBlock::text(""),
                            "thinking" => anthropic::ContentBlock::Thinking {
                                thinking: String::new(),
                                signature: None,
                            },
                            "tool_use" => anthropic::ContentBlock::ToolUse {
                                id: cb.id.clone().unwrap_or_default(),
                                name: cb.name.clone().unwrap_or_default(),
                                input: serde_json::json!({}),
                                cache_control: None,
                            },
                            _ => continue,
                        };
                        let out = anthropic::StreamEvent::ContentBlockStart {
                            index,
                            content_block,
                        };
                        return Some((Ok(out), (lines, model, state)));
                    }
                    "content_block_stop" => {
                        let index = event.index.unwrap_or(state.next_index.saturating_sub(1));
                        let out = anthropic::StreamEvent::ContentBlockStop { index };
                        return Some((Ok(out), (lines, model, state)));
                    }
                    "content_block_delta" => {
                        let Some(delta) = &event.delta else {
                            continue;
                        };
                        let index = event.index.unwrap_or(state.next_index.saturating_sub(1));
                        let block_delta = match delta.kind.as_str() {
                            "text_delta" => anthropic::BlockDelta::Text {
                                text: delta.text.clone(),
                            },
                            "thinking_delta" => anthropic::BlockDelta::Thinking {
                                thinking: delta
                                    .thinking
                                    .clone()
                                    .unwrap_or_else(|| delta.text.clone()),
                            },
                            "input_json_delta" => {
                                let Some(partial) = &delta.partial_json else {
                                    continue;
                                };
                                anthropic::BlockDelta::InputJson {
                                    partial_json: partial.clone(),
                                }
                            }
                            _ => continue,
                        };
                        let out = anthropic::StreamEvent::ContentBlockDelta {
                            index,
                            delta: block_delta,
                        };
                        return Some((Ok(out), (lines, model, state)));
                    }
                    "message_delta" => {
                        let stop_reason = event.delta.as_ref().and_then(|d| d.stop_reason.clone());
                        let usage = anthropic::Usage {
                            input_tokens: state.input_usage.input_tokens,
                            output_tokens: event
                                .usage
                                .as_ref()
                                .map(|u| u.output_tokens)
                                .unwrap_or(0),
                            cache_read_input_tokens: state.input_usage.cache_read_input_tokens,
                            cache_creation_input_tokens: state
                                .input_usage
                                .cache_creation_input_tokens,
                        };
                        let out = anthropic::StreamEvent::MessageDelta {
                            delta: anthropic::MessageDeltaPayload {
                                stop_reason,
                                stop_sequence: None,
                            },
                            usage,
                        };
                        return Some((Ok(out), (lines, model, state)));
                    }
                    "message_stop" => {
                        return Some((
                            Ok(anthropic::StreamEvent::MessageStop),
                            (lines, model, state),
                        ));
                    }
                    _ => {}
                }
            }
        },
    )
}

/// Convert a stream of native Anthropic events to OpenAI-shaped chunks.
pub fn anthropic_events_to_chunks(
    events: impl Stream<Item = Result<anthropic::StreamEvent, Error>> + Send + 'static,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> + Send + 'static {
    struct ChunkState {
        model: String,
        chunk_idx: u64,
        tool_call_idx: u32,
        input_usage: anthropic::Usage,
    }

    stream::unfold(
        (
            events.boxed(),
            ChunkState {
                model: String::new(),
                chunk_idx: 0,
                tool_call_idx: 0,
                input_usage: anthropic::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            },
        ),
        |(mut events, mut state)| async move {
            use futures_util::StreamExt;

            loop {
                let event = match events.next().await? {
                    Ok(e) => e,
                    Err(e) => return Some((Err(e), (events, state))),
                };

                match event {
                    anthropic::StreamEvent::MessageStart { message } => {
                        state.model = message.model;
                        state.input_usage = message.usage;
                    }
                    anthropic::StreamEvent::ContentBlockStart { content_block, .. } => {
                        if let anthropic::ContentBlock::ToolUse { id, name, .. } = &content_block {
                            state.chunk_idx += 1;
                            let tool_idx = state.tool_call_idx;
                            state.tool_call_idx += 1;
                            let chunk = ChatCompletionChunk {
                                id: format!("chatcmpl-{}", state.chunk_idx),
                                object: "chat.completion.chunk".to_string(),
                                created: 0,
                                model: state.model.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: if state.chunk_idx == 1 {
                                            Some(Role::Assistant)
                                        } else {
                                            None
                                        },
                                        content: None,
                                        tool_calls: Some(vec![ToolCallDelta {
                                            index: tool_idx,
                                            id: Some(id.clone()),
                                            kind: Some(ToolType::Function),
                                            function: Some(FunctionCallDelta {
                                                name: Some(name.clone()),
                                                arguments: Some(String::new()),
                                            }),
                                        }]),
                                        reasoning_content: None,
                                    },
                                    finish_reason: None,
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: None,
                            };
                            return Some((Ok(chunk), (events, state)));
                        }
                    }
                    anthropic::StreamEvent::ContentBlockDelta { delta, .. } => {
                        state.chunk_idx += 1;
                        let oai_delta = match delta {
                            anthropic::BlockDelta::Text { text } => Delta {
                                role: if state.chunk_idx == 1 {
                                    Some(Role::Assistant)
                                } else {
                                    None
                                },
                                content: Some(text),
                                tool_calls: None,
                                reasoning_content: None,
                            },
                            anthropic::BlockDelta::Thinking { thinking } => {
                                if thinking.is_empty() {
                                    continue;
                                }
                                Delta {
                                    role: if state.chunk_idx == 1 {
                                        Some(Role::Assistant)
                                    } else {
                                        None
                                    },
                                    content: None,
                                    tool_calls: None,
                                    reasoning_content: Some(thinking),
                                }
                            }
                            anthropic::BlockDelta::InputJson { partial_json } => Delta {
                                role: None,
                                content: None,
                                tool_calls: Some(vec![ToolCallDelta {
                                    index: state.tool_call_idx.saturating_sub(1),
                                    id: None,
                                    kind: None,
                                    function: Some(FunctionCallDelta {
                                        name: None,
                                        arguments: Some(partial_json),
                                    }),
                                }]),
                                reasoning_content: None,
                            },
                        };
                        let chunk = ChatCompletionChunk {
                            id: format!("chatcmpl-{}", state.chunk_idx),
                            object: "chat.completion.chunk".to_string(),
                            created: 0,
                            model: state.model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: oai_delta,
                                finish_reason: None,
                                logprobs: None,
                            }],
                            usage: None,
                            system_fingerprint: None,
                        };
                        return Some((Ok(chunk), (events, state)));
                    }
                    anthropic::StreamEvent::MessageDelta { delta, usage } => {
                        let finish_reason =
                            delta.stop_reason.as_deref().map(anthropic::finish_reason);
                        state.chunk_idx += 1;
                        let chunk = ChatCompletionChunk {
                            id: format!("chatcmpl-{}", state.chunk_idx),
                            object: "chat.completion.chunk".to_string(),
                            created: 0,
                            model: state.model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    tool_calls: None,
                                    reasoning_content: None,
                                },
                                finish_reason,
                                logprobs: None,
                            }],
                            usage: Some(OpenAiUsage::from(&Usage {
                                input_tokens: state.input_usage.input_tokens,
                                cache_read_tokens: state
                                    .input_usage
                                    .cache_read_input_tokens
                                    .unwrap_or(0),
                                cache_write_tokens: state
                                    .input_usage
                                    .cache_creation_input_tokens
                                    .unwrap_or(0),
                                output_tokens: usage.output_tokens,
                                reasoning_tokens: 0,
                                server_tool_calls: Default::default(),
                            })),
                            system_fingerprint: None,
                        };
                        return Some((Ok(chunk), (events, state)));
                    }
                    anthropic::StreamEvent::MessageStop => return None,
                    anthropic::StreamEvent::ContentBlockStop { .. } => {}
                }
            }
        },
    )
}

/// Convert an OpenAI-shaped chunk stream into native Anthropic streaming
/// events. Reconstructs block boundaries from the flat delta stream.
pub fn chunks_to_anthropic_events(
    chunks: impl Stream<Item = Result<ChatCompletionChunk, Error>> + Unpin + Send + 'static,
) -> impl Stream<Item = Result<anthropic::StreamEvent, Error>> + Send + 'static {
    use alloc::collections::VecDeque;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CurrentBlock {
        Text { index: u32 },
        Thinking { index: u32 },
        ToolUse { index: u32, openai_index: u32 },
    }

    struct State {
        started: bool,
        finished: bool,
        current: Option<CurrentBlock>,
        next_index: u32,
        pending: VecDeque<anthropic::StreamEvent>,
        deferred_error: Option<Error>,
        latest_usage: Option<OpenAiUsage>,
        stop_reason: Option<String>,
    }

    impl State {
        fn ensure_started(&mut self, chunk: &ChatCompletionChunk) {
            if self.started {
                return;
            }
            self.started = true;
            let msg = anthropic::Response {
                id: chunk.id.clone(),
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                model: chunk.model.clone(),
                content: Vec::new(),
                stop_reason: None,
                stop_sequence: None,
                usage: anthropic::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            };
            self.pending
                .push_back(anthropic::StreamEvent::MessageStart { message: msg });
        }

        fn close_current(&mut self) {
            if let Some(block) = self.current.take() {
                let index = match block {
                    CurrentBlock::Text { index }
                    | CurrentBlock::Thinking { index }
                    | CurrentBlock::ToolUse { index, .. } => index,
                };
                self.pending
                    .push_back(anthropic::StreamEvent::ContentBlockStop { index });
            }
        }

        fn switch_to_text(&mut self) -> u32 {
            if let Some(CurrentBlock::Text { index }) = self.current {
                return index;
            }
            self.close_current();
            let index = self.next_index;
            self.next_index += 1;
            self.pending
                .push_back(anthropic::StreamEvent::ContentBlockStart {
                    index,
                    content_block: anthropic::ContentBlock::text(""),
                });
            self.current = Some(CurrentBlock::Text { index });
            index
        }

        fn switch_to_thinking(&mut self) -> u32 {
            if let Some(CurrentBlock::Thinking { index }) = self.current {
                return index;
            }
            self.close_current();
            let index = self.next_index;
            self.next_index += 1;
            self.pending
                .push_back(anthropic::StreamEvent::ContentBlockStart {
                    index,
                    content_block: anthropic::ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: None,
                    },
                });
            self.current = Some(CurrentBlock::Thinking { index });
            index
        }

        fn open_tool_use(&mut self, openai_index: u32, id: String, name: String) -> u32 {
            self.close_current();
            let index = self.next_index;
            self.next_index += 1;
            self.pending
                .push_back(anthropic::StreamEvent::ContentBlockStart {
                    index,
                    content_block: anthropic::ContentBlock::ToolUse {
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
                    .push_back(anthropic::StreamEvent::ContentBlockDelta {
                        index,
                        delta: anthropic::BlockDelta::Thinking {
                            thinking: reasoning,
                        },
                    });
            }

            if let Some(text) = delta.content
                && !text.is_empty()
            {
                let index = self.switch_to_text();
                self.pending
                    .push_back(anthropic::StreamEvent::ContentBlockDelta {
                        index,
                        delta: anthropic::BlockDelta::Text { text },
                    });
            }

            if let Some(tool_deltas) = delta.tool_calls {
                for tc in tool_deltas {
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
                            .push_back(anthropic::StreamEvent::ContentBlockDelta {
                                index: current_index,
                                delta: anthropic::BlockDelta::InputJson { partial_json: args },
                            });
                    }
                }
            }

            if let Some(reason) = choice.finish_reason {
                self.stop_reason = Some(match reason {
                    FinishReason::Stop => "end_turn".to_string(),
                    FinishReason::Length => "max_tokens".to_string(),
                    FinishReason::ToolCalls => "tool_use".to_string(),
                    FinishReason::ContentFilter => "content_filter".to_string(),
                    FinishReason::Custom(s) => s,
                });
            }
        }

        fn finalize(&mut self, default_stop: String) {
            self.close_current();
            let stop_reason = self.stop_reason.take().or(Some(default_stop));
            let usage = self
                .latest_usage
                .take()
                .map(|u| anthropic::Usage::from(&Usage::from(&u)))
                .unwrap_or(anthropic::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                });
            self.pending
                .push_back(anthropic::StreamEvent::MessageDelta {
                    delta: anthropic::MessageDeltaPayload {
                        stop_reason,
                        stop_sequence: None,
                    },
                    usage,
                });
            self.pending.push_back(anthropic::StreamEvent::MessageStop);
            self.finished = true;
        }
    }

    let state = State {
        started: false,
        finished: false,
        current: None,
        next_index: 0,
        pending: VecDeque::new(),
        deferred_error: None,
        latest_usage: None,
        stop_reason: None,
    };

    stream::unfold(
        (chunks.boxed(), state),
        |(mut chunks, mut state)| async move {
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
                    Some(Err(e)) => {
                        if state.started {
                            state.finalize("error".to_string());
                        } else {
                            state.finished = true;
                        }
                        state.deferred_error = Some(e);
                    }
                    None => {
                        if state.started {
                            state.finalize("end_turn".to_string());
                        } else {
                            return None;
                        }
                    }
                }
            }
        },
    )
}

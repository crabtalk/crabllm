use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use bytes::{Buf, Bytes, BytesMut};
use crabllm_core::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicStreamEvent, AnthropicSystem, AnthropicTool, AnthropicUsage, BlockDelta, BoxStream,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChunkChoice, ContentBlock,
    DEFAULT_MAX_TOKENS, Delta, Error, FinishReason, FunctionCallDelta, MessageDeltaPayload,
    OpenAiUsage, Provider, Role, Stop, ThinkingConfig, ToolCallDelta, ToolChoice, ToolType, Usage,
};
use futures::stream::{self, Stream, StreamExt};
use serde::Deserialize;

const THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    pub(crate) client: HttpClient,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
}

impl Provider for AnthropicProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        chat_completion(&self.client, &self.base_url, &self.api_key, request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let mut anthropic_req = translate_request(request);
        anthropic_req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&anthropic_req)
            .map_err(|e| Error::Internal(e.to_string()))?;
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let auth = auth_headers(&self.api_key);
        let mut headers: Vec<(&str, &str)> = vec![
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ];
        for (k, v) in &auth {
            headers.push((k, v.as_str()));
        }
        if anthropic_req.thinking.is_some() {
            headers.push(("anthropic-beta", THINKING_BETA));
        }
        let byte_stream = self.client.post_stream(&url, &headers, body.into()).await?;
        let events = anthropic_event_stream(byte_stream, request.model.clone());
        Ok(anthropic_events_to_chunks(events).boxed())
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let body =
            crabllm_core::json::to_vec(request).map_err(|e| Error::Internal(e.to_string()))?;
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let auth = auth_headers(&self.api_key);
        let mut headers: Vec<(&str, &str)> = vec![
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ];
        for (k, v) in &auth {
            headers.push((k, v.as_str()));
        }
        if request.thinking.is_some() {
            headers.push(("anthropic-beta", THINKING_BETA));
        }
        let resp = self
            .client
            .post(&url, &headers, body.into())
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
        if resp.status >= 400 {
            let body = String::from_utf8_lossy(&resp.body).into_owned();
            return Err(Error::Provider {
                status: resp.status,
                body,
                retry_after: resp.retry_after,
            });
        }
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Internal(e.to_string()))?;
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let auth = auth_headers(&self.api_key);
        let mut headers: Vec<(&str, &str)> = vec![
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ];
        for (k, v) in &auth {
            headers.push((k, v.as_str()));
        }
        if request.thinking.is_some() {
            headers.push(("anthropic-beta", THINKING_BETA));
        }
        let byte_stream = self.client.post_stream(&url, &headers, body.into()).await?;
        Ok(anthropic_event_stream(byte_stream, request.model.clone()).boxed())
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &crabllm_core::GeminiRequest,
    ) -> Result<BoxStream<'static, Result<crabllm_core::GeminiResponse, Error>>, Error> {
        crate::gemini_stream_via_chat(self, model, request).await
    }

    async fn complete(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<crabllm_core::ir::Response, Error> {
        let native = crabllm_core::AnthropicRequest::from(request);
        let resp = self.anthropic_messages(&native).await?;
        Ok(crabllm_core::ir::Response::from(resp))
    }

    async fn complete_stream(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ir::StreamEvent, Error>>, Error> {
        let mut native = crabllm_core::AnthropicRequest::from(request);
        native.stream = Some(true);
        let stream = self.anthropic_messages_stream(&native).await?;
        Ok(stream
            .flat_map(|result| {
                let events: Vec<Result<crabllm_core::ir::StreamEvent, Error>> = match result {
                    Ok(event) => event.to_ir_events().into_iter().map(Ok).collect(),
                    Err(e) => vec![Err(e)],
                };
                futures::stream::iter(events)
            })
            .boxed())
    }

    fn is_anthropic_compat(&self) -> bool {
        true
    }

    async fn anthropic_messages_raw(&self, raw_body: Bytes) -> Result<Bytes, Error> {
        anthropic_messages_raw(&self.client, &self.base_url, &self.api_key, raw_body).await
    }

    async fn anthropic_messages_stream_raw(
        &self,
        raw_body: Bytes,
    ) -> Result<crabllm_core::ByteStream, Error> {
        anthropic_messages_stream(&self.client, &self.base_url, &self.api_key, raw_body).await
    }
}

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const OAUTH_TOKEN_PREFIX: &str = "sk-ant-oat";
const OAUTH_BETA: &str = "oauth-2025-04-20";

// ── Anthropic SSE event types (parse-only) ──

#[derive(Deserialize)]
struct SseEvent {
    #[serde(rename = "type")]
    kind: String,
    #[allow(dead_code)]
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Option<SseDelta>,
    #[serde(default)]
    content_block: Option<SseContentBlock>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    message: Option<SseMessage>,
    #[serde(default)]
    error: Option<SseError>,
}

#[derive(Deserialize)]
struct SseMessage {
    #[serde(default)]
    usage: Option<AnthropicUsage>,
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

// ── Translation ──

fn translate_request(request: &ChatCompletionRequest) -> AnthropicRequest {
    let mut system_blocks = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        if msg.role == Role::System {
            for block in &msg.content {
                if let ContentBlock::Text { .. } = block {
                    system_blocks.push(block.clone());
                }
            }
        } else {
            messages.push(AnthropicMessage {
                role: msg.role.as_str().to_string(),
                content: AnthropicContent::Blocks(msg.content.clone()),
            });
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else if system_blocks.iter().any(|b| {
        matches!(
            b,
            ContentBlock::Text {
                cache_control: Some(_),
                ..
            }
        )
    }) {
        Some(AnthropicSystem::Blocks(system_blocks))
    } else {
        let joined = system_blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Some(AnthropicSystem::Text(joined))
    };

    // B2: When tool_choice is "none", omit tools and tool_choice entirely.
    let is_none = request.tool_choice.as_ref() == Some(&ToolChoice::Disabled);

    let tools = if is_none {
        None
    } else {
        request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| AnthropicTool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: {
                        let mut s = t
                            .function
                            .parameters
                            .clone()
                            .unwrap_or(serde_json::json!({"type": "object"}));
                        schema::inline_refs(&mut s);
                        s
                    },
                })
                .collect()
        })
    };

    let tool_choice = if is_none {
        None
    } else {
        request.tool_choice.as_ref().map(|tc| match tc {
            ToolChoice::Auto => serde_json::json!({"type": "auto"}),
            ToolChoice::Required => serde_json::json!({"type": "any"}),
            ToolChoice::Function { name } => serde_json::json!({"type": "tool", "name": name}),
            ToolChoice::Disabled => unreachable!(),
        })
    };

    let stop_sequences = request.stop.as_ref().map(|s| match s {
        Stop::Single(s) => vec![s.clone()],
        Stop::Multiple(v) => v.clone(),
    });

    let max_tokens = request
        .anthropic_max_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);

    let thinking = request.thinking.clone().or_else(|| {
        request.extra.get("thinking").and_then(|v| {
            if v.as_bool() == Some(true) {
                Some(ThinkingConfig {
                    kind: "enabled".to_string(),
                    budget_tokens: Some(max_tokens.saturating_sub(1)),
                })
            } else if let Some(obj) = v.as_object() {
                let budget = obj
                    .get("budget_tokens")
                    .and_then(|b| b.as_u64())
                    .map(|b| b as u32);
                Some(ThinkingConfig {
                    kind: "enabled".to_string(),
                    budget_tokens: budget,
                })
            } else {
                None
            }
        })
    });

    AnthropicRequest {
        model: request.model.clone(),
        messages,
        max_tokens,
        system,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: request.stream,
        tools,
        tool_choice,
        stop_sequences,
        thinking,
    }
}

// ── Auth helpers ──

fn is_oauth_token(api_key: &str) -> bool {
    api_key.starts_with(OAUTH_TOKEN_PREFIX)
}

fn auth_headers(api_key: &str) -> Vec<(&'static str, String)> {
    if is_oauth_token(api_key) {
        vec![
            ("authorization", format!("Bearer {api_key}")),
            ("anthropic-beta", OAUTH_BETA.to_string()),
        ]
    } else {
        vec![("x-api-key", api_key.to_string())]
    }
}

// ── Public API ──

/// Forward raw Anthropic-format JSON bytes to the Messages API,
/// returning the response bytes without deserialization.
pub async fn anthropic_messages_raw(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<Bytes, Error> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let auth = auth_headers(api_key);
    let mut headers: Vec<(&str, &str)> = vec![
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ];
    for (k, v) in &auth {
        headers.push((k, v.as_str()));
    }
    let resp = client
        .post(&url, &headers, raw_body)
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
            retry_after: resp.retry_after,
        });
    }

    Ok(resp.body)
}

/// Stream raw Anthropic SSE bytes from the Messages API.
pub async fn anthropic_messages_stream(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<ByteStream, Error> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let auth = auth_headers(api_key);
    let mut headers: Vec<(&str, &str)> = vec![
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ];
    for (k, v) in &auth {
        headers.push((k, v.as_str()));
    }
    client.post_stream(&url, &headers, raw_body).await
}

pub async fn chat_completion(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    if is_oauth_token(api_key) {
        return Err(Error::Internal(
            "OAuth tokens only support streaming; set stream: true".into(),
        ));
    }

    let anthropic_req = translate_request(request);
    let url = format!("{}/messages", base_url.trim_end_matches('/'));

    let body =
        crabllm_core::json::to_vec(&anthropic_req).map_err(|e| Error::Internal(e.to_string()))?;
    let auth = auth_headers(api_key);
    let mut headers: Vec<(&str, &str)> = vec![
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ];
    for (k, v) in &auth {
        headers.push((k, v.as_str()));
    }
    if anthropic_req.thinking.is_some() {
        headers.push(("anthropic-beta", THINKING_BETA));
    }
    let resp = client
        .post(&url, &headers, body.into())
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
            retry_after: resp.retry_after,
        });
    }

    let anthropic_resp: AnthropicResponse =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))?;
    let ir_resp = crabllm_core::ir::Response::from(anthropic_resp);
    Ok(ChatCompletionResponse::from(&ir_resp))
}

/// Parse an Anthropic SSE byte stream into native `AnthropicStreamEvent`s.
pub fn anthropic_event_stream(
    byte_stream: ByteStream,
    model: String,
) -> impl Stream<Item = Result<AnthropicStreamEvent, Error>> {
    struct State {
        next_index: u32,
        input_usage: AnthropicUsage,
    }

    stream::unfold(
        (
            byte_stream,
            BytesMut::new(),
            model,
            State {
                next_index: 0,
                input_usage: AnthropicUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            },
        ),
        |(mut byte_stream, mut buffer, model, mut state)| async move {
            use futures::StreamExt;

            loop {
                if let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                    let mut line_end = newline_pos;
                    if line_end > 0 && buffer[line_end - 1] == b'\r' {
                        line_end -= 1;
                    }
                    let line = &buffer[..line_end];

                    if line.is_empty() {
                        buffer.advance(newline_pos + 1);
                        continue;
                    }

                    let Some(data) = line.strip_prefix(b"data: ") else {
                        buffer.advance(newline_pos + 1);
                        continue;
                    };
                    let Ok(data) = std::str::from_utf8(data) else {
                        buffer.advance(newline_pos + 1);
                        continue;
                    };
                    let data = data.trim();

                    let Ok(event) = crabllm_core::json::from_str::<SseEvent>(data) else {
                        buffer.advance(newline_pos + 1);
                        continue;
                    };
                    buffer.advance(newline_pos + 1);

                    match event.kind.as_str() {
                        "message_start" => {
                            if let Some(msg) = &event.message
                                && let Some(usage) = &msg.usage
                            {
                                state.input_usage = usage.clone();
                            }
                            let out = AnthropicStreamEvent::MessageStart {
                                message: AnthropicResponse {
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
                            return Some((Ok(out), (byte_stream, buffer, model, state)));
                        }
                        "error" => {
                            let msg = if let Some(err) = &event.error {
                                format!("anthropic stream error: {}: {}", err.kind, err.message)
                            } else {
                                "anthropic stream error: unknown".to_string()
                            };
                            return Some((
                                Err(Error::Internal(msg)),
                                (byte_stream, buffer, model, state),
                            ));
                        }
                        "content_block_start" => {
                            let Some(cb) = &event.content_block else {
                                continue;
                            };
                            let index = state.next_index;
                            state.next_index += 1;
                            let content_block = match cb.kind.as_str() {
                                "text" => AnthropicContentBlock::text(""),
                                "thinking" => AnthropicContentBlock::Thinking {
                                    thinking: String::new(),
                                    signature: None,
                                },
                                "tool_use" => AnthropicContentBlock::ToolUse {
                                    id: cb.id.clone().unwrap_or_default(),
                                    name: cb.name.clone().unwrap_or_default(),
                                    input: serde_json::json!({}),
                                    cache_control: None,
                                },
                                _ => continue,
                            };
                            let out = AnthropicStreamEvent::ContentBlockStart {
                                index,
                                content_block,
                            };
                            return Some((Ok(out), (byte_stream, buffer, model, state)));
                        }
                        "content_block_stop" => {
                            let index = event.index.unwrap_or(state.next_index.saturating_sub(1));
                            let out = AnthropicStreamEvent::ContentBlockStop { index };
                            return Some((Ok(out), (byte_stream, buffer, model, state)));
                        }
                        "content_block_delta" => {
                            let Some(delta) = &event.delta else {
                                continue;
                            };
                            let index = event.index.unwrap_or(state.next_index.saturating_sub(1));
                            let block_delta = match delta.kind.as_str() {
                                "text_delta" => BlockDelta::Text {
                                    text: delta.text.clone(),
                                },
                                "thinking_delta" => BlockDelta::Thinking {
                                    thinking: delta
                                        .thinking
                                        .clone()
                                        .unwrap_or_else(|| delta.text.clone()),
                                },
                                "input_json_delta" => {
                                    let Some(partial) = &delta.partial_json else {
                                        continue;
                                    };
                                    BlockDelta::InputJson {
                                        partial_json: partial.clone(),
                                    }
                                }
                                _ => continue,
                            };
                            let out = AnthropicStreamEvent::ContentBlockDelta {
                                index,
                                delta: block_delta,
                            };
                            return Some((Ok(out), (byte_stream, buffer, model, state)));
                        }
                        "message_delta" => {
                            let stop_reason =
                                event.delta.as_ref().and_then(|d| d.stop_reason.clone());
                            let usage = AnthropicUsage {
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
                            let out = AnthropicStreamEvent::MessageDelta {
                                delta: MessageDeltaPayload {
                                    stop_reason,
                                    stop_sequence: None,
                                },
                                usage,
                            };
                            return Some((Ok(out), (byte_stream, buffer, model, state)));
                        }
                        "message_stop" => {
                            return Some((
                                Ok(AnthropicStreamEvent::MessageStop),
                                (byte_stream, buffer, model, state),
                            ));
                        }
                        _ => {}
                    }
                    continue;
                }

                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.extend_from_slice(&bytes);
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buffer, model, state),
                        ));
                    }
                    None => return None,
                }
            }
        },
    )
}

/// Convert a stream of native Anthropic events to OpenAI-shaped chunks.
pub fn anthropic_events_to_chunks(
    events: impl Stream<Item = Result<AnthropicStreamEvent, Error>> + Send + 'static,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> + Send + 'static {
    struct ChunkState {
        model: String,
        chunk_idx: u64,
        tool_call_idx: u32,
        input_usage: AnthropicUsage,
    }

    stream::unfold(
        (
            events.boxed(),
            ChunkState {
                model: String::new(),
                chunk_idx: 0,
                tool_call_idx: 0,
                input_usage: AnthropicUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            },
        ),
        |(mut events, mut state)| async move {
            use futures::StreamExt;

            loop {
                let event = match events.next().await? {
                    Ok(e) => e,
                    Err(e) => return Some((Err(e), (events, state))),
                };

                match event {
                    AnthropicStreamEvent::MessageStart { message } => {
                        state.model = message.model;
                        state.input_usage = message.usage;
                    }
                    AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                        if let AnthropicContentBlock::ToolUse { id, name, .. } = &content_block {
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
                    AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                        state.chunk_idx += 1;
                        let oai_delta = match delta {
                            BlockDelta::Text { text } => Delta {
                                role: if state.chunk_idx == 1 {
                                    Some(Role::Assistant)
                                } else {
                                    None
                                },
                                content: Some(text),
                                tool_calls: None,
                                reasoning_content: None,
                            },
                            BlockDelta::Thinking { thinking } => {
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
                            BlockDelta::InputJson { partial_json } => Delta {
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
                    AnthropicStreamEvent::MessageDelta { delta, usage } => {
                        let finish_reason = delta.stop_reason.as_deref().map(|r| match r {
                            "end_turn" => FinishReason::Stop,
                            "max_tokens" => FinishReason::Length,
                            "tool_use" => FinishReason::ToolCalls,
                            other => FinishReason::Custom(other.to_string()),
                        });
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
                    AnthropicStreamEvent::MessageStop => return None,
                    AnthropicStreamEvent::ContentBlockStop { .. } => {}
                }
            }
        },
    )
}

/// Convert an OpenAI-shaped chunk stream into native Anthropic streaming
/// events. Reconstructs block boundaries from the flat delta stream.
pub fn chunks_to_anthropic_events(
    chunks: impl Stream<Item = Result<ChatCompletionChunk, Error>> + Unpin + Send + 'static,
) -> impl Stream<Item = Result<AnthropicStreamEvent, Error>> + Send + 'static {
    use std::collections::VecDeque;

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
        pending: VecDeque<AnthropicStreamEvent>,
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
            let msg = AnthropicResponse {
                id: chunk.id.clone(),
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                model: chunk.model.clone(),
                content: Vec::new(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            };
            self.pending
                .push_back(AnthropicStreamEvent::MessageStart { message: msg });
        }

        fn close_current(&mut self) {
            if let Some(block) = self.current.take() {
                let index = match block {
                    CurrentBlock::Text { index }
                    | CurrentBlock::Thinking { index }
                    | CurrentBlock::ToolUse { index, .. } => index,
                };
                self.pending
                    .push_back(AnthropicStreamEvent::ContentBlockStop { index });
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
                .push_back(AnthropicStreamEvent::ContentBlockStart {
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
            let index = self.next_index;
            self.next_index += 1;
            self.pending
                .push_back(AnthropicStreamEvent::ContentBlockStart {
                    index,
                    content_block: AnthropicContentBlock::Thinking {
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
                .push_back(AnthropicStreamEvent::ContentBlockStart {
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
                    .push_back(AnthropicStreamEvent::ContentBlockDelta {
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
                    .push_back(AnthropicStreamEvent::ContentBlockDelta {
                        index,
                        delta: BlockDelta::Text { text },
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
                            .push_back(AnthropicStreamEvent::ContentBlockDelta {
                                index: current_index,
                                delta: BlockDelta::InputJson { partial_json: args },
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
                .map(|u| AnthropicUsage::from(&Usage::from(&u)))
                .unwrap_or(AnthropicUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                });
            self.pending.push_back(AnthropicStreamEvent::MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason,
                    stop_sequence: None,
                },
                usage,
            });
            self.pending.push_back(AnthropicStreamEvent::MessageStop);
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

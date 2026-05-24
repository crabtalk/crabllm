use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use bytes::{Buf, Bytes, BytesMut};
use crabllm_core::{
    AnthropicContent, AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicSystem,
    AnthropicTool, AnthropicUsage, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, Choice, ChunkChoice, ContentBlock, DEFAULT_MAX_TOKENS, Delta, Error,
    FinishReason, FunctionCallDelta, Message, OpenAiUsage, Provider, Role, Stop, ThinkingConfig,
    ToolCallDelta, ToolChoice, ToolType, Usage,
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
        let s = chat_completion_stream(
            &self.client,
            &self.base_url,
            &self.api_key,
            request,
            &request.model,
        )
        .await?;
        Ok(s.boxed())
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
            });
        }
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
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
        Ok(anthropic_sse_stream(byte_stream, request.model.clone()).boxed())
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

fn map_usage(u: &AnthropicUsage) -> OpenAiUsage {
    OpenAiUsage::from(&Usage::from(u))
}

fn map_stop_reason(stop_reason: &Option<String>) -> Option<FinishReason> {
    stop_reason.as_ref().map(|r| match r.as_str() {
        "end_turn" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        other => FinishReason::Custom(other.to_string()),
    })
}

fn translate_response(resp: AnthropicResponse) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: resp.id,
        object: "chat.completion".to_string(),
        created: 0,
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: resp.content,
            },
            finish_reason: map_stop_reason(&resp.stop_reason),
            logprobs: None,
        }],
        usage: Some(map_usage(&resp.usage)),
        system_fingerprint: None,
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
        });
    }

    let anthropic_resp: AnthropicResponse =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))?;

    Ok(translate_response(anthropic_resp))
}

pub async fn chat_completion_stream(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    request: &ChatCompletionRequest,
    model: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let mut anthropic_req = translate_request(request);
    anthropic_req.stream = Some(true);
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
    let byte_stream = client.post_stream(&url, &headers, body.into()).await?;

    let model = model.to_string();
    Ok(anthropic_sse_stream(byte_stream, model))
}

/// Streaming state: tracks chunk counter, tool call counter, cached input tokens,
/// and whether the current content block is a thinking block.
struct StreamState {
    chunk_idx: u64,
    tool_call_idx: u32,
    input_tokens: u32,
    cache_read_input_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
    is_thinking_block: bool,
}

pub(crate) fn anthropic_sse_stream(
    byte_stream: ByteStream,
    model: String,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    let state = StreamState {
        chunk_idx: 0,
        tool_call_idx: 0,
        input_tokens: 0,
        cache_read_input_tokens: None,
        cache_creation_input_tokens: None,
        is_thinking_block: false,
    };

    stream::unfold(
        (byte_stream, BytesMut::new(), model, state),
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
                    let data = match std::str::from_utf8(data) {
                        Ok(s) => s.trim(),
                        Err(_) => {
                            buffer.advance(newline_pos + 1);
                            continue;
                        }
                    };

                    let event: SseEvent = match crabllm_core::json::from_str(data) {
                        Ok(e) => e,
                        Err(_) => {
                            buffer.advance(newline_pos + 1);
                            continue;
                        }
                    };
                    buffer.advance(newline_pos + 1);

                    match event.kind.as_str() {
                        "message_start" => {
                            if let Some(msg) = &event.message
                                && let Some(usage) = &msg.usage
                            {
                                state.input_tokens = usage.input_tokens;
                                state.cache_read_input_tokens = usage.cache_read_input_tokens;
                                state.cache_creation_input_tokens =
                                    usage.cache_creation_input_tokens;
                            }
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
                            match cb.kind.as_str() {
                                "thinking" => state.is_thinking_block = true,
                                "tool_use" => {
                                    state.is_thinking_block = false;
                                    state.chunk_idx += 1;
                                    let tool_idx = state.tool_call_idx;
                                    state.tool_call_idx += 1;
                                    let chunk = ChatCompletionChunk {
                                        id: format!("chatcmpl-{}", state.chunk_idx),
                                        object: "chat.completion.chunk".to_string(),
                                        created: 0,
                                        model: model.clone(),
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
                                                    id: cb.id.clone(),
                                                    kind: Some(ToolType::Function),
                                                    function: Some(FunctionCallDelta {
                                                        name: cb.name.clone(),
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
                                    return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                                }
                                _ => state.is_thinking_block = false,
                            }
                        }
                        "content_block_stop" => {
                            state.is_thinking_block = false;
                        }
                        "content_block_delta" => {
                            let Some(delta) = &event.delta else {
                                continue;
                            };
                            match delta.kind.as_str() {
                                "thinking_delta" => {
                                    let text = delta.thinking.as_deref().unwrap_or(&delta.text);
                                    if text.is_empty() {
                                        continue;
                                    }
                                    state.chunk_idx += 1;
                                    let chunk = ChatCompletionChunk {
                                        id: format!("chatcmpl-{}", state.chunk_idx),
                                        object: "chat.completion.chunk".to_string(),
                                        created: 0,
                                        model: model.clone(),
                                        choices: vec![ChunkChoice {
                                            index: 0,
                                            delta: Delta {
                                                role: if state.chunk_idx == 1 {
                                                    Some(Role::Assistant)
                                                } else {
                                                    None
                                                },
                                                content: None,
                                                tool_calls: None,
                                                reasoning_content: Some(text.to_string()),
                                            },
                                            finish_reason: None,
                                            logprobs: None,
                                        }],
                                        usage: None,
                                        system_fingerprint: None,
                                    };
                                    return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                                }
                                "text_delta" => {
                                    state.chunk_idx += 1;
                                    let chunk = ChatCompletionChunk {
                                        id: format!("chatcmpl-{}", state.chunk_idx),
                                        object: "chat.completion.chunk".to_string(),
                                        created: 0,
                                        model: model.clone(),
                                        choices: vec![ChunkChoice {
                                            index: 0,
                                            delta: Delta {
                                                role: if state.chunk_idx == 1 {
                                                    Some(Role::Assistant)
                                                } else {
                                                    None
                                                },
                                                content: Some(delta.text.clone()),
                                                tool_calls: None,
                                                reasoning_content: None,
                                            },
                                            finish_reason: None,
                                            logprobs: None,
                                        }],
                                        usage: None,
                                        system_fingerprint: None,
                                    };
                                    return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                                }
                                "input_json_delta" => {
                                    let Some(partial) = &delta.partial_json else {
                                        continue;
                                    };
                                    state.chunk_idx += 1;
                                    let tool_idx = state.tool_call_idx.saturating_sub(1);
                                    let chunk = ChatCompletionChunk {
                                        id: format!("chatcmpl-{}", state.chunk_idx),
                                        object: "chat.completion.chunk".to_string(),
                                        created: 0,
                                        model: model.clone(),
                                        choices: vec![ChunkChoice {
                                            index: 0,
                                            delta: Delta {
                                                role: None,
                                                content: None,
                                                tool_calls: Some(vec![ToolCallDelta {
                                                    index: tool_idx,
                                                    id: None,
                                                    kind: None,
                                                    function: Some(FunctionCallDelta {
                                                        name: None,
                                                        arguments: Some(partial.clone()),
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
                                    return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                                }
                                _ => {}
                            }
                        }
                        "message_delta" => {
                            let Some(delta) = &event.delta else {
                                continue;
                            };
                            let finish_reason = map_stop_reason(&delta.stop_reason);
                            state.chunk_idx += 1;
                            let chunk = ChatCompletionChunk {
                                id: format!("chatcmpl-{}", state.chunk_idx),
                                object: "chat.completion.chunk".to_string(),
                                created: 0,
                                model: model.clone(),
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
                                usage: event.usage.map(|u| {
                                    OpenAiUsage::from(&Usage {
                                        input_tokens: state.input_tokens,
                                        cache_read_tokens: state
                                            .cache_read_input_tokens
                                            .unwrap_or(0),
                                        cache_write_tokens: state
                                            .cache_creation_input_tokens
                                            .unwrap_or(0),
                                        output_tokens: u.output_tokens,
                                        reasoning_tokens: 0,
                                        server_tool_calls: Default::default(),
                                    })
                                }),
                                system_fingerprint: None,
                            };
                            return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                        }
                        "message_stop" => return None,
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

use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, Delta,
    Error, FunctionCall, FunctionCallDelta, Message, ToolCall, ToolCallDelta, Usage,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const BASE_URL: &str = "https://api.anthropic.com/v1";

// ── Anthropic-native request types ──

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

/// Message content: either a plain string or an array of content blocks.
#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// A content block in a message (text, tool_use, or tool_result).
#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: serde_json::Value,
}

// ── Anthropic-native response types ──

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ResponseContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ResponseContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── Anthropic SSE event types ──

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
    let mut system_parts = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content
                && let Some(s) = content.as_str()
            {
                system_parts.push(s.to_string());
            }
        } else if msg.role == "tool" {
            // Tool result → user message with tool_result content block.
            let content_str = msg
                .content
                .as_ref()
                .map(|c| {
                    if let Some(s) = c.as_str() {
                        s.to_string()
                    } else {
                        c.to_string()
                    }
                })
                .unwrap_or_default();
            let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content: content_str,
                }]),
            });
        } else if msg.role == "assistant"
            && let Some(tool_calls) = &msg.tool_calls
        {
            // Assistant message with tool_calls → content blocks.
            let mut blocks = Vec::new();
            if let Some(content) = &msg.content
                && let Some(s) = content.as_str()
                && !s.is_empty()
            {
                blocks.push(AnthropicContentBlock::Text {
                    text: s.to_string(),
                });
            }
            for tc in tool_calls {
                let input = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                blocks.push(AnthropicContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    input,
                });
            }
            messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: AnthropicContent::Blocks(blocks),
            });
        } else {
            let content = msg
                .content
                .as_ref()
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            messages.push(AnthropicMessage {
                role: msg.role.clone(),
                content: AnthropicContent::Text(content),
            });
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or(serde_json::json!({"type": "object"})),
            })
            .collect()
    });

    let tool_choice = request.tool_choice.as_ref().map(|tc| {
        if let Some(s) = tc.as_str() {
            match s {
                "auto" | "none" => serde_json::json!({"type": "auto"}),
                "required" => serde_json::json!({"type": "any"}),
                _ => serde_json::json!({"type": "auto"}),
            }
        } else if let Some(name) = tc.get("function").and_then(|f| f.get("name")) {
            serde_json::json!({"type": "tool", "name": name})
        } else {
            serde_json::json!({"type": "auto"})
        }
    });

    AnthropicRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: request.stream,
        tools,
        tool_choice,
    }
}

fn map_stop_reason(stop_reason: &Option<String>) -> Option<String> {
    stop_reason.as_ref().map(|r| match r.as_str() {
        "end_turn" => "stop".to_string(),
        "max_tokens" => "length".to_string(),
        "tool_use" => "tool_calls".to_string(),
        other => other.to_string(),
    })
}

fn translate_response(resp: AnthropicResponse) -> ChatCompletionResponse {
    let mut content_text = String::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block.kind.as_str() {
            "text" => content_text.push_str(&block.text),
            "tool_use" => {
                if let (Some(id), Some(name), Some(input)) = (&block.id, &block.name, &block.input)
                {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        kind: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    });
                }
            }
            _ => {}
        }
    }

    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let content = if content_text.is_empty() && tool_calls_opt.is_some() {
        None
    } else {
        Some(serde_json::Value::String(content_text))
    };

    ChatCompletionResponse {
        id: resp.id,
        object: "chat.completion".to_string(),
        created: 0,
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content,
                tool_calls: tool_calls_opt,
                tool_call_id: None,
                name: None,
            },
            finish_reason: map_stop_reason(&resp.stop_reason),
        }],
        usage: Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
    }
}

pub fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("anthropic {name} not supported"))
}

// ── Public API ──

pub async fn chat_completion(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let anthropic_req = translate_request(request);
    let url = format!("{BASE_URL}/messages");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&anthropic_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let anthropic_resp: AnthropicResponse = resp
        .json()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    Ok(translate_response(anthropic_resp))
}

pub async fn chat_completion_stream(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
    model: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let mut anthropic_req = translate_request(request);
    anthropic_req.stream = Some(true);
    let url = format!("{BASE_URL}/messages");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&anthropic_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let model = model.to_string();
    Ok(anthropic_sse_stream(resp, model))
}

/// Streaming state: tracks chunk counter and tool call counter for indexing.
struct StreamState {
    chunk_idx: u64,
    tool_call_idx: u32,
}

fn anthropic_sse_stream(
    resp: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    let byte_stream = resp.bytes_stream();
    let state = StreamState {
        chunk_idx: 0,
        tool_call_idx: 0,
    };

    stream::unfold(
        (byte_stream, String::new(), model, state),
        |(mut byte_stream, mut buffer, model, mut state)| async move {
            use futures::TryStreamExt;

            loop {
                if let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    let data = data.trim();

                    let event: SseEvent = match serde_json::from_str(data) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    match event.kind.as_str() {
                        "content_block_start" => {
                            // Tool use block start: emit initial ToolCallDelta.
                            if let Some(cb) = &event.content_block
                                && cb.kind == "tool_use"
                            {
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
                                                Some("assistant".to_string())
                                            } else {
                                                None
                                            },
                                            content: None,
                                            tool_calls: Some(vec![ToolCallDelta {
                                                index: tool_idx,
                                                id: cb.id.clone(),
                                                kind: Some("function".to_string()),
                                                function: Some(FunctionCallDelta {
                                                    name: cb.name.clone(),
                                                    arguments: Some(String::new()),
                                                }),
                                            }]),
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = &event.delta {
                                if delta.kind == "text_delta" {
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
                                                    Some("assistant".to_string())
                                                } else {
                                                    None
                                                },
                                                content: Some(delta.text.clone()),
                                                tool_calls: None,
                                            },
                                            finish_reason: None,
                                        }],
                                        usage: None,
                                    };
                                    return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                                } else if delta.kind == "input_json_delta" {
                                    // Tool input delta: emit ToolCallDelta with partial JSON.
                                    if let Some(partial) = &delta.partial_json {
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
                                                },
                                                finish_reason: None,
                                            }],
                                            usage: None,
                                        };
                                        return Some((
                                            Ok(chunk),
                                            (byte_stream, buffer, model, state),
                                        ));
                                    }
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(delta) = &event.delta {
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
                                        },
                                        finish_reason,
                                    }],
                                    usage: event.usage.map(|u| Usage {
                                        prompt_tokens: u.input_tokens,
                                        completion_tokens: u.output_tokens,
                                        total_tokens: u.input_tokens + u.output_tokens,
                                    }),
                                };
                                return Some((Ok(chunk), (byte_stream, buffer, model, state)));
                            }
                        }
                        "message_stop" => return None,
                        _ => continue,
                    }
                }

                match byte_stream.try_next().await {
                    Ok(Some(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Ok(None) => return None,
                    Err(e) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buffer, model, state),
                        ));
                    }
                }
            }
        },
    )
}

use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, Delta,
    Error, FunctionCall, FunctionCallDelta, Message, ToolCall, ToolCallDelta, Usage,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ── Gemini-native request types ──

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiToolDef>>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}

#[derive(Serialize, Deserialize, Clone)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolDef {
    function_declarations: Vec<GeminiFunctionDecl>,
}

#[derive(Serialize)]
struct GeminiFunctionDecl {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

// ── Gemini-native response types ──

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
}

// ── Translation ──

fn translate_request(request: &ChatCompletionRequest) -> GeminiRequest {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content
                && let Some(s) = content.as_str()
            {
                system_parts.push(GeminiPart {
                    text: Some(s.to_string()),
                    function_call: None,
                    function_response: None,
                });
            }
        } else if msg.role == "tool" {
            // Tool result → user message with functionResponse part.
            let name = msg.name.clone().unwrap_or_default();
            let response_val = msg
                .content
                .as_ref()
                .and_then(|c| {
                    if c.is_object() || c.is_array() {
                        Some(c.clone())
                    } else if let Some(s) = c.as_str() {
                        serde_json::from_str(s).ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(serde_json::json!({"result": msg.content.as_ref().and_then(|c| c.as_str()).unwrap_or("")}));
            contents.push(GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart {
                    text: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse {
                        name,
                        response: response_val,
                    }),
                }],
            });
        } else if msg.role == "assistant"
            && let Some(tool_calls) = &msg.tool_calls
        {
            // Assistant message with tool_calls → model message with functionCall parts.
            let mut parts = Vec::new();
            if let Some(content) = &msg.content
                && let Some(s) = content.as_str()
                && !s.is_empty()
            {
                parts.push(GeminiPart {
                    text: Some(s.to_string()),
                    function_call: None,
                    function_response: None,
                });
            }
            for tc in tool_calls {
                let args = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                parts.push(GeminiPart {
                    text: None,
                    function_call: Some(GeminiFunctionCall {
                        name: tc.function.name.clone(),
                        args,
                    }),
                    function_response: None,
                });
            }
            contents.push(GeminiContent {
                role: Some("model".to_string()),
                parts,
            });
        } else {
            let role = match msg.role.as_str() {
                "assistant" => "model",
                other => other,
            };
            let text = msg
                .content
                .as_ref()
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            contents.push(GeminiContent {
                role: Some(role.to_string()),
                parts: vec![GeminiPart {
                    text: Some(text),
                    function_call: None,
                    function_response: None,
                }],
            });
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: None,
            parts: system_parts,
        })
    };

    let stop_sequences = request.stop.as_ref().map(|s| match s {
        crabtalk_core::Stop::Single(s) => vec![s.clone()],
        crabtalk_core::Stop::Multiple(v) => v.clone(),
    });

    let generation_config = Some(GenerationConfig {
        max_output_tokens: Some(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences,
    });

    let tools = request.tools.as_ref().map(|tools| {
        vec![GeminiToolDef {
            function_declarations: tools
                .iter()
                .map(|t| GeminiFunctionDecl {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                })
                .collect(),
        }]
    });

    GeminiRequest {
        contents,
        system_instruction,
        generation_config,
        tools,
    }
}

fn map_finish_reason(reason: &Option<String>) -> Option<String> {
    reason.as_ref().map(|r| match r.as_str() {
        "STOP" => "stop".to_string(),
        "MAX_TOKENS" => "length".to_string(),
        "SAFETY" => "content_filter".to_string(),
        other => other.to_lowercase(),
    })
}

/// Extract text and tool calls from response candidate parts.
fn extract_parts(candidate: &GeminiCandidate) -> (String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = &candidate.content {
        for (i, part) in content.parts.iter().enumerate() {
            if let Some(t) = &part.text {
                text.push_str(t);
            }
            if let Some(fc) = &part.function_call {
                tool_calls.push(ToolCall {
                    id: format!("call_{i}"),
                    kind: "function".to_string(),
                    function: FunctionCall {
                        name: fc.name.clone(),
                        arguments: serde_json::to_string(&fc.args).unwrap_or_default(),
                    },
                });
            }
        }
    }

    (text, tool_calls)
}

fn translate_response(resp: GeminiResponse, model: &str) -> ChatCompletionResponse {
    let (content_text, tool_calls, finish_reason) = resp
        .candidates
        .first()
        .map(|c| {
            let (text, tcs) = extract_parts(c);
            (text, tcs, map_finish_reason(&c.finish_reason))
        })
        .unwrap_or_default();

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
        id: String::new(),
        object: "chat.completion".to_string(),
        created: 0,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content,
                tool_calls: tool_calls_opt,
                tool_call_id: None,
                name: None,
            },
            finish_reason,
        }],
        usage: resp.usage_metadata.map(|u| Usage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        }),
    }
}

pub fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("google {name} not supported"))
}

// ── Public API ──

pub async fn chat_completion(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let gemini_req = translate_request(request);
    let url = format!("{}/models/{}:generateContent", BASE_URL, request.model);

    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&gemini_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let gemini_resp: GeminiResponse = resp
        .json()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    Ok(translate_response(gemini_resp, &request.model))
}

pub async fn chat_completion_stream(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
    model: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let gemini_req = translate_request(request);
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        BASE_URL, request.model
    );

    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&gemini_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let model = model.to_string();
    Ok(gemini_sse_stream(resp, model))
}

fn gemini_sse_stream(
    resp: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    let byte_stream = resp.bytes_stream();

    stream::unfold(
        (byte_stream, String::new(), model, 0u64),
        |(mut byte_stream, mut buffer, model, mut chunk_idx)| async move {
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

                    let gemini_resp: GeminiResponse = match serde_json::from_str(data) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };

                    let candidate = match gemini_resp.candidates.first() {
                        Some(c) => c,
                        None => continue,
                    };

                    let (text, tool_calls) = extract_parts(candidate);
                    let finish_reason = map_finish_reason(&candidate.finish_reason);

                    let has_text = !text.is_empty();
                    let has_tools = !tool_calls.is_empty();

                    // Skip chunks with no content, no tool calls, and no finish reason.
                    if !has_text && !has_tools && finish_reason.is_none() {
                        continue;
                    }

                    chunk_idx += 1;
                    let tool_call_deltas = if has_tools {
                        Some(
                            tool_calls
                                .into_iter()
                                .enumerate()
                                .map(|(i, tc)| ToolCallDelta {
                                    index: i as u32,
                                    id: Some(tc.id),
                                    kind: Some("function".to_string()),
                                    function: Some(FunctionCallDelta {
                                        name: Some(tc.function.name),
                                        arguments: Some(tc.function.arguments),
                                    }),
                                })
                                .collect(),
                        )
                    } else {
                        None
                    };

                    let chunk = ChatCompletionChunk {
                        id: format!("chatcmpl-{chunk_idx}"),
                        object: "chat.completion.chunk".to_string(),
                        created: 0,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: if chunk_idx == 1 {
                                    Some("assistant".to_string())
                                } else {
                                    None
                                },
                                content: if has_text { Some(text) } else { None },
                                tool_calls: tool_call_deltas,
                            },
                            finish_reason,
                        }],
                        usage: gemini_resp.usage_metadata.map(|u| Usage {
                            prompt_tokens: u.prompt_token_count,
                            completion_tokens: u.candidates_token_count,
                            total_tokens: u.total_token_count,
                        }),
                    };
                    return Some((Ok(chunk), (byte_stream, buffer, model, chunk_idx)));
                }

                match byte_stream.try_next().await {
                    Ok(Some(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Ok(None) => return None,
                    Err(e) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buffer, model, chunk_idx),
                        ));
                    }
                }
            }
        },
    )
}

use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use bytes::{Buf, BytesMut};
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, ContentBlock, Delta, Error,
    FunctionCallDelta, GeminiCandidate, GeminiContent, GeminiFunctionCall, GeminiFunctionDecl,
    GeminiFunctionResponse, GeminiPart, GeminiRequest, GeminiResponse, GeminiRole, GeminiToolDef,
    GenerationConfig, Message, OpenAiUsage, Provider, Role, ToolCallDelta, ToolResultContent,
    Usage,
};
use futures::stream::{self, Stream, StreamExt};

#[derive(Debug, Clone)]
pub struct GoogleProvider {
    pub(crate) client: HttpClient,
    pub(crate) api_key: String,
}

impl Provider for GoogleProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let gemini_req = translate_request(request);
        let url = format!("{BASE_URL}/models/{}:generateContent", request.model);
        let body =
            crabllm_core::json::to_vec(&gemini_req).map_err(|e| Error::Encode(e.to_string()))?;
        let headers = [
            ("x-goog-api-key", self.api_key.as_str()),
            ("content-type", "application/json"),
        ];
        let resp = self.client.post(&url, &headers, body.into()).await?;
        if resp.status >= 400 {
            return Err(Error::Provider {
                status: resp.status,
                body: String::from_utf8_lossy(&resp.body).into_owned(),
                retry_after: resp.retry_after,
            });
        }
        let gemini_resp: GeminiResponse =
            crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))?;
        Ok(translate_response(gemini_resp, &request.model))
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let gemini_req = translate_request(request);
        let url = format!(
            "{BASE_URL}/models/{}:streamGenerateContent?alt=sse",
            request.model
        );
        let body =
            crabllm_core::json::to_vec(&gemini_req).map_err(|e| Error::Encode(e.to_string()))?;
        let headers = [
            ("x-goog-api-key", self.api_key.as_str()),
            ("content-type", "application/json"),
        ];
        let byte_stream = self.client.post_stream(&url, &headers, body.into()).await?;
        let model = request.model.clone();
        Ok(gemini_responses_to_chunks(gemini_event_stream(byte_stream), model).boxed())
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let ir_resp = self
            .complete(&crabllm_core::ir::Request::from(request.clone()))
            .await?;
        Ok(AnthropicResponse::from(&ir_resp))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        crate::anthropic_stream_via_chat(self, request).await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
        let url = format!("{BASE_URL}/models/{model}:streamGenerateContent?alt=sse");
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let headers = [
            ("x-goog-api-key", self.api_key.as_str()),
            ("content-type", "application/json"),
        ];
        let byte_stream = self.client.post_stream(&url, &headers, body.into()).await?;
        Ok(gemini_event_stream(byte_stream).boxed())
    }

    async fn complete(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<crabllm_core::ir::Response, Error> {
        let native = ChatCompletionRequest::from(request);
        let resp = self.chat_completion(&native).await?;
        Ok(crabllm_core::ir::Response::from(resp))
    }

    async fn complete_stream(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ir::StreamEvent, Error>>, Error> {
        let mut native = ChatCompletionRequest::from(request);
        native.stream = Some(true);
        let stream = self.chat_completion_stream(&native).await?;
        Ok(stream
            .flat_map(|result| {
                let events: Vec<Result<crabllm_core::ir::StreamEvent, Error>> = match result {
                    Ok(chunk) => chunk.to_ir_events().into_iter().map(Ok).collect(),
                    Err(e) => vec![Err(e)],
                };
                futures::stream::iter(events)
            })
            .boxed())
    }

    fn is_gemini_compat(&self) -> bool {
        true
    }

    async fn gemini_generate_content_raw(
        &self,
        model: &str,
        raw_body: bytes::Bytes,
    ) -> Result<bytes::Bytes, Error> {
        let url = format!("{BASE_URL}/models/{model}:generateContent");
        let headers = [
            ("x-goog-api-key", self.api_key.as_str()),
            ("content-type", "application/json"),
        ];
        let resp = self.client.post(&url, &headers, raw_body).await?;
        if resp.status >= 400 {
            return Err(Error::Provider {
                status: resp.status,
                body: String::from_utf8_lossy(&resp.body).into_owned(),
                retry_after: resp.retry_after,
            });
        }
        Ok(resp.body)
    }

    async fn gemini_generate_content_stream_raw(
        &self,
        model: &str,
        raw_body: bytes::Bytes,
    ) -> Result<ByteStream, Error> {
        let url = format!("{BASE_URL}/models/{model}:streamGenerateContent?alt=sse");
        let headers = [
            ("x-goog-api-key", self.api_key.as_str()),
            ("content-type", "application/json"),
        ];
        self.client.post_stream(&url, &headers, raw_body).await
    }
}

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// Gemini 2.5+ thinking models attach a `thoughtSignature` (base64) to
// each `functionCall` part. Subsequent turns must echo that signature
// back or the API returns 400. Since crabllm presents the OpenAI shape
// to downstream callers — no provider-specific fields, no SDK
// cooperation — we encode the signature into the synthetic
// `tool_call.id` so it round-trips for free. Same approach as LiteLLM
// (`__thought__` separator) and Bifrost (`_ts_` separator). Pick the
// LiteLLM separator: collision-resistant against well-formed tool ids.
//
// See: https://ai.google.dev/gemini-api/docs/thought-signatures
const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

/// Sentinel asking Gemini to skip signature validation. Required for
/// gemini-3+ when no real signature is available (e.g. fresh
/// conversation, or the previous turn predated this feature). Mirrors
/// LiteLLM's "dummy signature" / Bifrost's `skip_thought_signature_validator`.
const SKIP_VALIDATOR_SIGNATURE: &str = "skip_thought_signature_validator";

fn encode_signature_into_id(base_id: &str, signature: Option<&str>) -> String {
    match signature {
        Some(sig) if !sig.is_empty() => {
            format!("{base_id}{THOUGHT_SIGNATURE_SEPARATOR}{sig}")
        }
        _ => base_id.to_string(),
    }
}

fn extract_signature_from_id(tool_call_id: &str) -> Option<&str> {
    tool_call_id
        .split_once(THOUGHT_SIGNATURE_SEPARATOR)
        .map(|(_, sig)| sig)
}

fn is_gemini_3_or_newer(model: &str) -> bool {
    // "gemini-3", "gemini-3-pro", "gemini-3.5-...", and any later major
    // version. Match the prefix only — variant suffixes aren't relevant.
    let m = model.to_ascii_lowercase();
    let Some(rest) = m.strip_prefix("gemini-") else {
        return false;
    };
    // Read leading digits; a major version >= 3 means signature-required.
    let major_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    major_str.parse::<u32>().map(|n| n >= 3).unwrap_or(false)
}

// ── Translation ──

fn translate_request(request: &ChatCompletionRequest) -> GeminiRequest {
    // Build tool_use id → name index so ToolResult blocks can resolve the
    // function name for Gemini's functionResponse.
    let mut tc_names = std::collections::HashMap::<&str, &str>::new();
    for msg in &request.messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                tc_names.insert(id.as_str(), name.as_str());
            }
        }
    }

    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let needs_skip_sentinel = is_gemini_3_or_newer(&request.model);

    for msg in &request.messages {
        if msg.role == Role::System {
            for block in &msg.content {
                if let ContentBlock::Text { text, .. } = block {
                    system_parts.push(GeminiPart {
                        text: Some(text.clone()),
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                    });
                }
            }
        } else {
            let role = match msg.role {
                Role::Assistant => GeminiRole::Model,
                _ => GeminiRole::User,
            };
            let mut parts = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text, .. } if !text.is_empty() => {
                        parts.push(GeminiPart {
                            text: Some(text.clone()),
                            function_call: None,
                            function_response: None,
                            thought_signature: None,
                        });
                    }
                    ContentBlock::ToolUse {
                        id, name, input, ..
                    } => {
                        let signature = extract_signature_from_id(id)
                            .map(|s| s.to_string())
                            .or_else(|| {
                                needs_skip_sentinel.then(|| SKIP_VALIDATOR_SIGNATURE.to_string())
                            });
                        parts.push(GeminiPart {
                            text: None,
                            function_call: Some(GeminiFunctionCall {
                                name: name.clone(),
                                args: input.clone(),
                            }),
                            function_response: None,
                            thought_signature: signature,
                        });
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        name,
                        content,
                        ..
                    } => {
                        let fn_name = name
                            .as_deref()
                            .or_else(|| tc_names.get(tool_use_id.as_str()).copied())
                            .unwrap_or("")
                            .to_string();
                        let text = match content {
                            ToolResultContent::Text(s) => s.clone(),
                            ToolResultContent::Blocks(blocks) => blocks
                                .iter()
                                .filter_map(|b| match b {
                                    ContentBlock::Text { text, .. } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        let response_val = crabllm_core::json::from_str(&text)
                            .unwrap_or(serde_json::json!({"result": text}));
                        parts.push(GeminiPart {
                            text: None,
                            function_call: None,
                            function_response: Some(GeminiFunctionResponse {
                                name: fn_name,
                                response: response_val,
                            }),
                            thought_signature: None,
                        });
                    }
                    _ => {}
                }
            }
            if !parts.is_empty() {
                contents.push(GeminiContent {
                    role: Some(role),
                    parts,
                });
            }
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
        crabllm_core::Stop::Single(s) => vec![s.clone()],
        crabllm_core::Stop::Multiple(v) => v.clone(),
    });

    let generation_config = Some(GenerationConfig {
        max_output_tokens: request.anthropic_max_tokens.or(request.max_tokens),
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
                    parameters: t.function.parameters.clone().map(|mut p| {
                        schema::inline_refs(&mut p);
                        schema::strip_schema_meta(&mut p);
                        schema::flatten_nullable(&mut p);
                        schema::strip_fields(
                            &mut p,
                            &[
                                "title",
                                "default",
                                "examples",
                                "$comment",
                                "additionalProperties",
                            ],
                        );
                        p
                    }),
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

fn translate_response(resp: GeminiResponse, model: &str) -> ChatCompletionResponse {
    let (blocks, finish_reason) = resp
        .candidates
        .first()
        .map(|c| {
            (
                candidate_to_blocks(c),
                c.finish_reason.as_ref().map(Into::into),
            )
        })
        .unwrap_or_default();

    ChatCompletionResponse {
        id: String::new(),
        object: "chat.completion".to_string(),
        created: 0,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: blocks,
            },
            finish_reason,
            logprobs: None,
        }],
        usage: resp
            .usage_metadata
            .as_ref()
            .map(|u| OpenAiUsage::from(&Usage::from(u))),
        system_fingerprint: None,
    }
}

fn candidate_to_blocks(candidate: &GeminiCandidate) -> Vec<ContentBlock> {
    let Some(content) = &candidate.content else {
        return Vec::new();
    };
    let mut blocks = Vec::new();
    for (i, part) in content.parts.iter().enumerate() {
        if let Some(t) = &part.text
            && !t.is_empty()
        {
            blocks.push(ContentBlock::text(t.clone()));
        }
        if let Some(fc) = &part.function_call {
            let base_id = format!("call_{i}");
            let id = encode_signature_into_id(&base_id, part.thought_signature.as_deref());
            blocks.push(ContentBlock::ToolUse {
                id,
                name: fc.name.clone(),
                input: fc.args.clone(),
                cache_control: None,
            });
        }
    }
    blocks
}

/// Parse a Gemini SSE byte stream into native `GeminiResponse` items.
///
/// Each `data:` line carries a full `GeminiResponse` JSON object. Chunks
/// with no candidates are skipped. The stream terminates when the
/// upstream closes.
pub fn gemini_event_stream(
    byte_stream: ByteStream,
) -> impl Stream<Item = Result<GeminiResponse, Error>> {
    stream::unfold(
        (byte_stream, BytesMut::new()),
        |(mut byte_stream, mut buffer)| async move {
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

                    let Ok(gemini_resp) = crabllm_core::json::from_str::<GeminiResponse>(data)
                    else {
                        buffer.advance(newline_pos + 1);
                        continue;
                    };

                    if gemini_resp.candidates.is_empty() {
                        buffer.advance(newline_pos + 1);
                        continue;
                    }

                    buffer.advance(newline_pos + 1);
                    return Some((Ok(gemini_resp), (byte_stream, buffer)));
                }

                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.extend_from_slice(&bytes);
                    }
                    Some(Err(e)) => {
                        return Some((Err(Error::Network(e.to_string())), (byte_stream, buffer)));
                    }
                    None => return None,
                }
            }
        },
    )
}

/// Convert a stream of native `GeminiResponse` items into OpenAI-shaped
/// `ChatCompletionChunk` items. Inverse of [`chunks_to_gemini_responses`].
pub fn gemini_responses_to_chunks(
    responses: impl Stream<Item = Result<GeminiResponse, Error>> + Send + 'static,
    model: String,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> + Send + 'static {
    stream::unfold(
        (responses.boxed(), model, 0u64),
        |(mut responses, model, mut chunk_idx)| async move {
            use futures::StreamExt;

            loop {
                let gemini_resp = match responses.next().await? {
                    Ok(r) => r,
                    Err(e) => return Some((Err(e), (responses, model, chunk_idx))),
                };

                let Some(candidate) = gemini_resp.candidates.first() else {
                    continue;
                };

                let blocks = candidate_to_blocks(candidate);
                let finish_reason = candidate.finish_reason.as_ref().map(Into::into);

                let mut text = String::new();
                let mut tool_call_deltas: Vec<ToolCallDelta> = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text: t, .. } => text.push_str(&t),
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
                            tool_call_deltas.push(ToolCallDelta {
                                index: tool_call_deltas.len() as u32,
                                id: Some(id),
                                kind: Some(crabllm_core::ToolType::Function),
                                function: Some(FunctionCallDelta {
                                    name: Some(name),
                                    arguments: Some(
                                        crabllm_core::json::to_string(&input).unwrap_or_default(),
                                    ),
                                }),
                            });
                        }
                        _ => {}
                    }
                }

                let has_text = !text.is_empty();
                let has_tools = !tool_call_deltas.is_empty();

                if !has_text && !has_tools && finish_reason.is_none() {
                    continue;
                }

                chunk_idx += 1;
                let tool_call_deltas = if has_tools {
                    Some(tool_call_deltas)
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
                                Some(Role::Assistant)
                            } else {
                                None
                            },
                            content: if has_text { Some(text) } else { None },
                            tool_calls: tool_call_deltas,
                            reasoning_content: None,
                        },
                        finish_reason,
                        logprobs: None,
                    }],
                    usage: gemini_resp
                        .usage_metadata
                        .as_ref()
                        .map(|u| OpenAiUsage::from(&Usage::from(u))),
                    system_fingerprint: None,
                };
                return Some((Ok(chunk), (responses, model, chunk_idx)));
            }
        },
    )
}

/// Convert an OpenAI-shaped `ChatCompletionChunk` stream into native
/// `GeminiResponse` items. Each chunk maps 1:1 to one response.
pub fn chunks_to_gemini_responses(
    chunks: impl Stream<Item = Result<ChatCompletionChunk, Error>> + Send + 'static,
) -> impl Stream<Item = Result<GeminiResponse, Error>> + Send + 'static {
    use crabllm_core::{GeminiFinishReason, GeminiUsage};

    chunks.map(|result| {
        result.map(|chunk| {
            let choice = chunk.choices.into_iter().next();
            let (finish_reason, parts) = match choice {
                Some(c) => {
                    let fr = c.finish_reason.as_ref().map(GeminiFinishReason::from);
                    let mut parts = Vec::new();
                    if let Some(text) = c.delta.content {
                        parts.push(GeminiPart {
                            text: Some(text),
                            function_call: None,
                            function_response: None,
                            thought_signature: None,
                        });
                    }
                    if let Some(tool_calls) = c.delta.tool_calls {
                        for tc in tool_calls {
                            let name = tc
                                .function
                                .as_ref()
                                .and_then(|f| f.name.clone())
                                .unwrap_or_default();
                            let args_str = tc
                                .function
                                .as_ref()
                                .and_then(|f| f.arguments.clone())
                                .unwrap_or_default();
                            let args: serde_json::Value = crabllm_core::json::from_str(&args_str)
                                .unwrap_or(serde_json::json!({}));
                            parts.push(GeminiPart {
                                text: None,
                                function_call: Some(GeminiFunctionCall { name, args }),
                                function_response: None,
                                thought_signature: None,
                            });
                        }
                    }
                    (fr, parts)
                }
                None => (None, Vec::new()),
            };

            let usage_metadata = chunk.usage.as_ref().map(|u| {
                let canonical = Usage::from(u);
                GeminiUsage::from(&canonical)
            });

            let candidate = GeminiCandidate {
                content: if parts.is_empty() {
                    None
                } else {
                    Some(GeminiContent {
                        role: Some(GeminiRole::Model),
                        parts,
                    })
                },
                finish_reason,
            };

            GeminiResponse {
                candidates: vec![candidate],
                usage_metadata,
            }
        })
    })
}

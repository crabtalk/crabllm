use crate::{
    ByteStream, ChatCompletionChunk, ChunkChoice, ContentBlock, Delta, Error, FunctionCallDelta,
    GeminiCandidate, GeminiContent, GeminiFunctionCall, GeminiPart, GeminiResponse, GeminiRole,
    OpenAiUsage, Role, ToolCallDelta, Usage,
};
use futures::stream::{self, Stream, StreamExt};

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
pub const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

pub fn encode_signature_into_id(base_id: &str, signature: Option<&str>) -> String {
    match signature {
        Some(sig) if !sig.is_empty() => {
            format!("{base_id}{THOUGHT_SIGNATURE_SEPARATOR}{sig}")
        }
        _ => base_id.to_string(),
    }
}

pub fn extract_signature_from_id(tool_call_id: &str) -> Option<&str> {
    tool_call_id
        .split_once(THOUGHT_SIGNATURE_SEPARATOR)
        .map(|(_, sig)| sig)
}

pub fn candidate_to_blocks(candidate: &GeminiCandidate) -> Vec<ContentBlock> {
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
    crate::codec::sse::data_lines(byte_stream).filter_map(|line| async move {
        match line {
            Err(e) => Some(Err(e)),
            // Unparsable payloads and candidate-less chunks are skipped.
            Ok(data) => match crate::json::from_str::<GeminiResponse>(&data) {
                Ok(resp) if !resp.candidates.is_empty() => Some(Ok(resp)),
                _ => None,
            },
        }
    })
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
                                kind: Some(crate::ToolType::Function),
                                function: Some(FunctionCallDelta {
                                    name: Some(name),
                                    arguments: Some(
                                        crate::json::to_string(&input).unwrap_or_default(),
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
    use crate::{GeminiFinishReason, GeminiUsage};

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
                            let args: serde_json::Value =
                                crate::json::from_str(&args_str).unwrap_or(serde_json::json!({}));
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

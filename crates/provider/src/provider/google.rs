use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use crabllm_core::codec::gemini::{
    candidate_to_blocks, extract_signature_from_id, gemini_event_stream, gemini_responses_to_chunks,
};
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, Choice, ContentBlock, Error, GeminiContent,
    GeminiFunctionCall, GeminiFunctionDecl, GeminiFunctionResponse, GeminiPart, GeminiRequest,
    GeminiResponse, GeminiRole, GeminiToolDef, GenerationConfig, Message, OpenAiUsage, Provider,
    Role, ToolResultContent, Usage,
};
use futures::StreamExt;

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

/// Sentinel asking Gemini to skip signature validation. Required for
/// gemini-3+ when no real signature is available (e.g. fresh
/// conversation, or the previous turn predated this feature). Mirrors
/// LiteLLM's "dummy signature" / Bifrost's `skip_thought_signature_validator`.
const SKIP_VALIDATOR_SIGNATURE: &str = "skip_thought_signature_validator";

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

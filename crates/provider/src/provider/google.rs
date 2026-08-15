use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use crabllm_core::codec::gemini::{
    candidate_to_blocks, extract_signature_from_id, gemini_event_stream, gemini_responses_to_chunks,
};
use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ContentBlock, Error, Message, Model, ModelList, OpenAiUsage, Provider, Role, ToolResultContent,
    Usage, anthropic, gemini,
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
        let resp = self
            .client
            .post(&url, &headers, body.into())
            .await?
            .error_for_status()?;
        let gemini_resp: gemini::Response =
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
        request: &anthropic::Request,
    ) -> Result<anthropic::Response, Error> {
        let ir_resp = self
            .complete(&crabllm_core::ir::Request::from(request.clone()))
            .await?;
        Ok(anthropic::Response::from(&ir_resp))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &anthropic::Request,
    ) -> Result<BoxStream<'static, Result<anthropic::StreamEvent, Error>>, Error> {
        crate::anthropic_stream_via_chat(self, request).await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &gemini::Request,
    ) -> Result<BoxStream<'static, Result<gemini::Response, Error>>, Error> {
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
        crate::complete_via_chat(self, request).await
    }

    async fn complete_stream(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ir::StreamEvent, Error>>, Error> {
        crate::complete_stream_via_chat(self, request).await
    }

    fn is_gemini_compat(&self) -> bool {
        true
    }

    async fn models(&self) -> Result<ModelList, Error> {
        models(&self.client, &self.api_key).await
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
        let resp = self
            .client
            .post(&url, &headers, raw_body)
            .await?
            .error_for_status()?;
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

/// Gemini's list rows carry the id as a resource name — `models/gemini-3-pro`
/// — while every request path spells it bare.
#[derive(serde::Deserialize)]
struct ModelsResponse {
    models: Vec<ModelEntry>,
}

#[derive(serde::Deserialize)]
struct ModelEntry {
    name: String,
}

/// List the models the key grants. Pages at 50 by default, 1000 max.
pub async fn models(client: &HttpClient, api_key: &str) -> Result<ModelList, Error> {
    let url = format!("{BASE_URL}/models?pageSize=1000");
    let headers = [("x-goog-api-key", api_key)];
    let resp = client.get(&url, &headers).await?.error_for_status()?;

    let list: ModelsResponse =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))?;
    Ok(ModelList {
        object: "list".to_string(),
        data: list
            .models
            .into_iter()
            .map(|m| Model {
                owned_by: "google".to_string(),
                ..Model::new(m.name.trim_start_matches("models/"))
            })
            .collect(),
    })
}

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

fn translate_request(request: &ChatCompletionRequest) -> gemini::Request {
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
                    system_parts.push(gemini::Part {
                        text: Some(text.clone()),
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                    });
                }
            }
        } else {
            let role = match msg.role {
                Role::Assistant => gemini::Role::Model,
                _ => gemini::Role::User,
            };
            let mut parts = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text, .. } if !text.is_empty() => {
                        parts.push(gemini::Part {
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
                        parts.push(gemini::Part {
                            text: None,
                            function_call: Some(gemini::FunctionCall {
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
                        parts.push(gemini::Part {
                            text: None,
                            function_call: None,
                            function_response: Some(gemini::FunctionResponse {
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
                contents.push(gemini::Content {
                    role: Some(role),
                    parts,
                });
            }
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(gemini::Content {
            role: None,
            parts: system_parts,
        })
    };

    let stop_sequences = request.stop.as_ref().map(|s| match s {
        crabllm_core::Stop::Single(s) => vec![s.clone()],
        crabllm_core::Stop::Multiple(v) => v.clone(),
    });

    let generation_config = Some(gemini::GenerationConfig {
        max_output_tokens: request.anthropic_max_tokens.or(request.max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences,
    });

    let tools = request.tools.as_ref().map(|tools| {
        vec![gemini::ToolDef {
            function_declarations: tools
                .iter()
                .map(|t| gemini::FunctionDecl {
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

    gemini::Request {
        contents,
        system_instruction,
        generation_config,
        tools,
    }
}

fn translate_response(resp: gemini::Response, model: &str) -> ChatCompletionResponse {
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

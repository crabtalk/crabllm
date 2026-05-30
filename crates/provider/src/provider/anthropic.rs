use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use bytes::Bytes;
use crabllm_core::{
    AnthropicContent, AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicStreamEvent,
    AnthropicSystem, AnthropicTool, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ContentBlock, DEFAULT_MAX_TOKENS, Error, Provider, Role, Stop,
    ThinkingConfig, ToolChoice,
    codec::anthropic::{anthropic_event_stream, anthropic_events_to_chunks},
};
use futures::stream::StreamExt;

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
        let body =
            crabllm_core::json::to_vec(&anthropic_req).map_err(|e| Error::Encode(e.to_string()))?;
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
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
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
        let resp = self.client.post(&url, &headers, body.into()).await?;
        if resp.status >= 400 {
            let body = String::from_utf8_lossy(&resp.body).into_owned();
            return Err(Error::Provider {
                status: resp.status,
                body,
                retry_after: resp.retry_after,
            });
        }
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Encode(e.to_string()))?;
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
                    cache_control: None,
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
    let resp = client.post(&url, &headers, raw_body).await?;

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
        return Err(Error::Invalid(
            "OAuth tokens only support streaming; set stream: true".into(),
        ));
    }

    let anthropic_req = translate_request(request);
    let url = format!("{}/messages", base_url.trim_end_matches('/'));

    let body =
        crabllm_core::json::to_vec(&anthropic_req).map_err(|e| Error::Encode(e.to_string()))?;
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
    let resp = client.post(&url, &headers, body.into()).await?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
            retry_after: resp.retry_after,
        });
    }

    let anthropic_resp: AnthropicResponse =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))?;
    let ir_resp = crabllm_core::ir::Response::from(anthropic_resp);
    Ok(ChatCompletionResponse::from(&ir_resp))
}

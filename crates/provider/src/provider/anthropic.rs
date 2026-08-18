use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use bytes::Bytes;
use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Error, Model,
    ModelList, Provider, ToolChoice, anthropic,
    anthropic::ContentBlock,
    codec::anthropic::{anthropic_event_stream, anthropic_events_to_chunks},
    ir,
};
use futures::stream::StreamExt;

/// Interleaved thinking for the manual `budget_tokens` dialect. Adaptive
/// thinking enables it on its own, so only the native passthrough — where
/// the client owns the request shape — still needs the header.
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
            ("anthropic-version", crabllm_core::anthropic::VERSION),
            ("content-type", "application/json"),
        ];
        for (k, v) in &auth {
            headers.push((k, v.as_str()));
        }
        let byte_stream = self.client.post_stream(&url, &headers, body.into()).await?;
        let events = anthropic_event_stream(byte_stream, request.model.clone());
        Ok(anthropic_events_to_chunks(events).boxed())
    }

    async fn anthropic_messages(
        &self,
        request: &anthropic::Request,
    ) -> Result<anthropic::Response, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let auth = auth_headers(&self.api_key);
        let mut headers: Vec<(&str, &str)> = vec![
            ("anthropic-version", crabllm_core::anthropic::VERSION),
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
            .await?
            .error_for_status()?;
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &anthropic::Request,
    ) -> Result<BoxStream<'static, Result<anthropic::StreamEvent, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Encode(e.to_string()))?;
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let auth = auth_headers(&self.api_key);
        let mut headers: Vec<(&str, &str)> = vec![
            ("anthropic-version", crabllm_core::anthropic::VERSION),
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
        request: &crabllm_core::gemini::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::gemini::Response, Error>>, Error> {
        crate::gemini_stream_via_chat(self, model, request).await
    }

    async fn complete(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<crabllm_core::ir::Response, Error> {
        let native = crabllm_core::anthropic::Request::from(request);
        let resp = self.anthropic_messages(&native).await?;
        Ok(crabllm_core::ir::Response::from(resp))
    }

    async fn complete_stream(
        &self,
        request: &crabllm_core::ir::Request,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ir::StreamEvent, Error>>, Error> {
        let mut native = crabllm_core::anthropic::Request::from(request);
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

    async fn models(&self) -> Result<ModelList, Error> {
        models(&self.client, &self.base_url, &self.api_key).await
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

fn translate_request(request: &ChatCompletionRequest) -> anthropic::Request {
    // The IR already knows how to read an OpenAI request and write an
    // Anthropic one, including tool-result coalescing and pairing. Only the
    // rules below are Anthropic-API quirks the IR has no reason to carry.
    let mut ir_req = ir::Request::from(request.clone());

    // `reasoning_effort` already arrived through the IR. This is the other
    // spelling: a raw Anthropic `thinking` object riding along in an
    // OpenAI-shaped body, which `extra` captures verbatim. Setting it on the
    // IR keeps it behind the same lowering as every other spelling.
    if let Some(v) = request.extra.get("thinking") {
        ir_req.thinking = if v.as_bool() == Some(true) {
            // A bare `true` used to mean "budget everything the response
            // allows"; `Max` is that intent in the dialect we now emit.
            Some(ir::Effort::Max)
        } else {
            v.as_object().and_then(|obj| {
                obj.get("budget_tokens")
                    .and_then(|b| b.as_u64())
                    .map(|b| ir::Effort::from(b as u32))
            })
        };
    }

    let mut out = anthropic::Request::from(&ir_req);

    // Anthropic rejects `tool_choice: "none"` sent alongside a tools array, so
    // when the choice is "none" we omit both tools and tool_choice entirely.
    if request.tool_choice.as_ref() == Some(&ToolChoice::Disabled) {
        out.tools = None;
        out.tool_choice = None;
    } else if let Some(tools) = out.tools.as_mut() {
        for tool in tools {
            schema::inline_refs(&mut tool.input_schema);
        }
    }

    // A single text system block goes as a bare string — the shape Anthropic
    // documents, and the one that avoids a needless array in the common case.
    if let Some(anthropic::System::Blocks(blocks)) = &out.system {
        let joined = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        out.system = Some(anthropic::System::Text(joined));
    }

    let max_tokens = request
        .anthropic_max_tokens
        .or(request.max_tokens)
        .unwrap_or(anthropic::DEFAULT_MAX_TOKENS);
    out.max_tokens = max_tokens;

    out.stream = request.stream;
    out
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

/// Anthropic's list rows: `{type, id, display_name, created_at}`. Only the id
/// survives translation — `created_at` is RFC 3339 where canonical `created`
/// is an epoch, and the gateway already emits 0 there.
#[derive(serde::Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(serde::Deserialize)]
struct ModelEntry {
    id: String,
}

/// List the models the key grants. The endpoint pages at 20 by default, so
/// ask for the documented maximum rather than silently truncating.
pub async fn models(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
) -> Result<ModelList, Error> {
    let url = format!("{}/models?limit=1000", base_url.trim_end_matches('/'));
    let auth = auth_headers(api_key);
    let mut headers: Vec<(&str, &str)> =
        vec![("anthropic-version", crabllm_core::anthropic::VERSION)];
    for (k, v) in &auth {
        headers.push((k, v.as_str()));
    }
    let resp = client.get(&url, &headers).await?.error_for_status()?;

    let list: ModelsResponse =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))?;
    Ok(ModelList {
        object: "list".to_string(),
        data: list
            .data
            .into_iter()
            .map(|m| Model {
                owned_by: "anthropic".to_string(),
                ..Model::new(m.id)
            })
            .collect(),
    })
}

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
        ("anthropic-version", crabllm_core::anthropic::VERSION),
        ("content-type", "application/json"),
    ];
    for (k, v) in &auth {
        headers.push((k, v.as_str()));
    }
    let resp = client
        .post(&url, &headers, raw_body)
        .await?
        .error_for_status()?;

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
        ("anthropic-version", crabllm_core::anthropic::VERSION),
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
        ("anthropic-version", crabllm_core::anthropic::VERSION),
        ("content-type", "application/json"),
    ];
    for (k, v) in &auth {
        headers.push((k, v.as_str()));
    }
    let resp = client
        .post(&url, &headers, body.into())
        .await?
        .error_for_status()?;

    let anthropic_resp: anthropic::Response =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))?;
    let ir_resp = crabllm_core::ir::Response::from(anthropic_resp);
    Ok(ChatCompletionResponse::from(&ir_resp))
}

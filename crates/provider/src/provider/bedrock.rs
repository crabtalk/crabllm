use crate::provider::schema;
use crate::{ByteStream, HttpClient};
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice,
    ContentBlock as CoreContentBlock, Delta, Error, FinishReason, FunctionCallDelta, Message, Role,
    ToolCallDelta, ToolType, Usage,
};
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};

pub(crate) use self::sigv4::sign_headers;

const BASE_URL: &str = "https://bedrock-runtime";
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct BedrockProvider {
    pub(crate) client: HttpClient,
    pub(crate) region: String,
    pub(crate) access_key: String,
    pub(crate) secret_key: String,
}

impl crabllm_core::Provider for BedrockProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        chat_completion(
            &self.client,
            &self.region,
            &self.access_key,
            &self.secret_key,
            request,
        )
        .await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<crabllm_core::BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let s = chat_completion_stream(
            &self.client,
            &self.region,
            &self.access_key,
            &self.secret_key,
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
        let ir_resp = self.complete(&crabllm_core::ir::Request::from(request.clone())).await?;
        Ok(AnthropicResponse::from(&ir_resp))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<crabllm_core::BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        crate::anthropic_stream_via_chat(self, request).await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &crabllm_core::GeminiRequest,
    ) -> Result<crabllm_core::BoxStream<'static, Result<crabllm_core::GeminiResponse, Error>>, Error> {
        crate::gemini_stream_via_chat(self, model, request).await
    }
}

// ── Bedrock Converse request types ──

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConverseRequest {
    messages: Vec<ConverseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_config: Option<InferenceConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
}

#[derive(Serialize)]
struct ConverseMessage {
    role: String,
    content: Vec<ContentBlock>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ContentBlock {
    Text(String),
    ToolUse {
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ToolResultContent {
    Text(String),
}

#[derive(Serialize)]
struct SystemBlock {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolConfig {
    tools: Vec<ToolDef>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolDef {
    tool_spec: ToolSpec,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolSpec {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: InputSchema,
}

#[derive(Serialize)]
struct InputSchema {
    json: serde_json::Value,
}

// ── Bedrock Converse response types ──

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConverseResponse {
    output: ConverseOutput,
    stop_reason: Option<String>,
    usage: Option<ConverseUsage>,
}

#[derive(Deserialize)]
struct ConverseOutput {
    message: Option<ConverseOutputMessage>,
}

#[derive(Deserialize)]
struct ConverseOutputMessage {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConverseUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

// ── Translation ──

fn translate_request(request: &ChatCompletionRequest) -> ConverseRequest {
    let mut system_blocks = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        if msg.role == Role::System {
            for block in &msg.content {
                if let CoreContentBlock::Text { text, .. } = block {
                    system_blocks.push(SystemBlock { text: text.clone() });
                }
            }
        } else {
            let blocks: Vec<ContentBlock> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    CoreContentBlock::Text { text, .. } => Some(ContentBlock::Text(text.clone())),
                    CoreContentBlock::ToolUse {
                        id, name, input, ..
                    } => Some(ContentBlock::ToolUse {
                        tool_use_id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    CoreContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        let text = match content {
                            crabllm_core::ToolResultContent::Text(s) => s.clone(),
                            crabllm_core::ToolResultContent::Blocks(blocks) => blocks
                                .iter()
                                .filter_map(|b| match b {
                                    CoreContentBlock::Text { text, .. } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        Some(ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: vec![ToolResultContent::Text(text)],
                        })
                    }
                    _ => None,
                })
                .collect();
            messages.push(ConverseMessage {
                role: msg.role.as_str().to_string(),
                content: blocks,
            });
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(system_blocks)
    };

    let stop_sequences = request.stop.as_ref().map(|s| match s {
        crabllm_core::Stop::Single(s) => vec![s.clone()],
        crabllm_core::Stop::Multiple(v) => v.clone(),
    });

    let inference_config = Some(InferenceConfig {
        max_tokens: Some(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences,
    });

    let tool_config = request.tools.as_ref().map(|tools| ToolConfig {
        tools: tools
            .iter()
            .map(|t| ToolDef {
                tool_spec: ToolSpec {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: InputSchema {
                        json: {
                            let mut s = t
                                .function
                                .parameters
                                .clone()
                                .unwrap_or(serde_json::json!({"type": "object"}));
                            schema::inline_refs(&mut s);
                            s
                        },
                    },
                },
            })
            .collect(),
    });

    ConverseRequest {
        messages,
        system,
        inference_config,
        tool_config,
    }
}

fn map_stop_reason(stop_reason: &Option<String>) -> Option<FinishReason> {
    stop_reason.as_ref().map(|r| match r.as_str() {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        other => FinishReason::Custom(other.to_string()),
    })
}

fn translate_response(resp: ConverseResponse, model: &str) -> ChatCompletionResponse {
    let mut blocks = Vec::new();

    if let Some(message) = &resp.output.message {
        for block in &message.content {
            match block {
                ContentBlock::Text(t) => {
                    blocks.push(CoreContentBlock::text(t.clone()));
                }
                ContentBlock::ToolUse {
                    tool_use_id,
                    name,
                    input,
                } => {
                    blocks.push(CoreContentBlock::ToolUse {
                        id: tool_use_id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                        cache_control: None,
                    });
                }
                ContentBlock::ToolResult { .. } => {}
            }
        }
    }

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
            finish_reason: map_stop_reason(&resp.stop_reason),
            logprobs: None,
        }],
        usage: resp.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
            ..Default::default()
        }),
        system_fingerprint: None,
    }
}

// ── Public API ──

pub async fn chat_completion(
    client: &HttpClient,
    region: &str,
    access_key: &str,
    secret_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let bedrock_req = translate_request(request);
    let body =
        crabllm_core::json::to_vec(&bedrock_req).map_err(|e| Error::Internal(e.to_string()))?;
    let url = format!(
        "{BASE_URL}.{region}.amazonaws.com/model/{}/converse",
        request.model
    );

    let signed = sign_headers("POST", &url, &body, region, access_key, secret_key)?;
    let headers: Vec<(&str, &str)> = signed
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
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

    let bedrock_resp: ConverseResponse =
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Internal(e.to_string()))?;

    Ok(translate_response(bedrock_resp, &request.model))
}

// ── Streaming types ──

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StreamEvent {
    #[serde(default)]
    message_start: Option<MessageStartEvent>,
    #[serde(default)]
    content_block_start: Option<ContentBlockStartEvent>,
    #[serde(default)]
    content_block_delta: Option<ContentBlockDeltaEvent>,
    #[serde(default)]
    message_stop: Option<MessageStopEvent>,
    #[serde(default)]
    metadata: Option<MetadataEvent>,
}

#[derive(Deserialize)]
struct MessageStartEvent {
    role: Role,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockStartEvent {
    start: Option<BlockStart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockStart {
    #[serde(default)]
    tool_use_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockDeltaEvent {
    delta: Option<BlockDelta>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockDelta {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    tool_use: Option<ToolUseDelta>,
}

#[derive(Deserialize)]
struct ToolUseDelta {
    input: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageStopEvent {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct MetadataEvent {
    usage: Option<ConverseUsage>,
}

// ── Event-stream binary frame parser ──

fn parse_event_payload(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    if buf.len() < 12 {
        return None;
    }

    let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < total_len {
        return None;
    }

    let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let payload_start = 12 + headers_len;
    let payload_end = total_len - 4;

    let payload = if payload_start < payload_end {
        buf[payload_start..payload_end].to_vec()
    } else {
        Vec::new()
    };

    buf.drain(..total_len);
    Some(payload)
}

// ── Streaming public API ──

pub async fn chat_completion_stream(
    client: &HttpClient,
    region: &str,
    access_key: &str,
    secret_key: &str,
    request: &ChatCompletionRequest,
    model: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let bedrock_req = translate_request(request);
    let body =
        crabllm_core::json::to_vec(&bedrock_req).map_err(|e| Error::Internal(e.to_string()))?;
    let url = format!(
        "{BASE_URL}.{region}.amazonaws.com/model/{}/converse-stream",
        request.model
    );

    let signed = sign_headers("POST", &url, &body, region, access_key, secret_key)?;
    let headers: Vec<(&str, &str)> = signed
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let byte_stream = client.post_stream(&url, &headers, body.into()).await?;

    let model = model.to_string();
    Ok(bedrock_event_stream(byte_stream, model))
}

struct BedrockStreamState {
    chunk_idx: u64,
    tool_call_idx: u32,
}

impl BedrockStreamState {
    fn next_chunk(&mut self, model: &str) -> ChatCompletionChunk {
        self.chunk_idx += 1;
        ChatCompletionChunk {
            id: format!("chatcmpl-{}", self.chunk_idx),
            object: "chat.completion.chunk".to_string(),
            model: model.to_string(),
            ..Default::default()
        }
    }
}

fn bedrock_event_stream(
    byte_stream: ByteStream,
    model: String,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    let state = BedrockStreamState {
        chunk_idx: 0,
        tool_call_idx: 0,
    };

    stream::unfold(
        (byte_stream, Vec::<u8>::new(), model, state),
        |(mut byte_stream, mut buf, model, mut state)| async move {
            use futures::StreamExt;

            loop {
                if let Some(payload) = parse_event_payload(&mut buf) {
                    if payload.is_empty() {
                        continue;
                    }

                    let event: StreamEvent = match crabllm_core::json::from_slice(&payload) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    if let Some(ms) = event.message_start {
                        let mut chunk = state.next_chunk(&model);
                        chunk.choices = vec![ChunkChoice {
                            delta: Delta {
                                role: Some(ms.role),
                                ..Default::default()
                            },
                            ..Default::default()
                        }];
                        return Some((Ok(chunk), (byte_stream, buf, model, state)));
                    }

                    if let Some(cbs) = event.content_block_start {
                        if let Some(start) = &cbs.start
                            && start.tool_use_id.is_some()
                        {
                            let tool_idx = state.tool_call_idx;
                            state.tool_call_idx += 1;
                            let mut chunk = state.next_chunk(&model);
                            chunk.choices = vec![ChunkChoice {
                                delta: Delta {
                                    tool_calls: Some(vec![ToolCallDelta {
                                        index: tool_idx,
                                        id: start.tool_use_id.clone(),
                                        kind: Some(ToolType::Function),
                                        function: Some(FunctionCallDelta {
                                            name: start.name.clone(),
                                            arguments: Some(String::new()),
                                        }),
                                    }]),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }];
                            return Some((Ok(chunk), (byte_stream, buf, model, state)));
                        }
                        continue;
                    }

                    if let Some(cbd) = event.content_block_delta {
                        if let Some(delta) = &cbd.delta {
                            if let Some(text) = &delta.text {
                                let mut chunk = state.next_chunk(&model);
                                chunk.choices = vec![ChunkChoice {
                                    delta: Delta {
                                        content: Some(text.clone()),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }];
                                return Some((Ok(chunk), (byte_stream, buf, model, state)));
                            }
                            if let Some(tu) = &delta.tool_use {
                                let tool_idx = state.tool_call_idx.saturating_sub(1);
                                let mut chunk = state.next_chunk(&model);
                                chunk.choices = vec![ChunkChoice {
                                    delta: Delta {
                                        tool_calls: Some(vec![ToolCallDelta {
                                            index: tool_idx,
                                            id: None,
                                            kind: None,
                                            function: Some(FunctionCallDelta {
                                                name: None,
                                                arguments: Some(tu.input.clone()),
                                            }),
                                        }]),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }];
                                return Some((Ok(chunk), (byte_stream, buf, model, state)));
                            }
                        }
                        continue;
                    }

                    if let Some(ms) = event.message_stop {
                        let mut chunk = state.next_chunk(&model);
                        chunk.choices = vec![ChunkChoice {
                            finish_reason: map_stop_reason(&ms.stop_reason),
                            ..Default::default()
                        }];
                        return Some((Ok(chunk), (byte_stream, buf, model, state)));
                    }

                    if let Some(meta) = event.metadata {
                        if let Some(u) = meta.usage {
                            let mut chunk = state.next_chunk(&model);
                            chunk.usage = Some(Usage {
                                prompt_tokens: u.input_tokens,
                                completion_tokens: u.output_tokens,
                                total_tokens: u.total_tokens,
                                ..Default::default()
                            });
                            return Some((Ok(chunk), (byte_stream, buf, model, state)));
                        }
                        return None;
                    }

                    continue;
                }

                match byte_stream.next().await {
                    Some(Ok(bytes)) => buf.extend_from_slice(&bytes),
                    Some(Err(e)) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buf, model, state),
                        ));
                    }
                    None => return None,
                }
            }
        },
    )
}

// ── SigV4 signing ──

mod sigv4 {
    use hmac::{Hmac, KeyInit, Mac};
    use sha2::{Digest, Sha256};
    use std::{
        fmt::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    const SERVICE: &str = "bedrock-runtime";

    pub fn sign_headers(
        method: &str,
        url: &str,
        body: &[u8],
        region: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Result<Vec<(String, String)>, crabllm_core::Error> {
        let parsed: http::Uri = url.parse().map_err(|e: http::uri::InvalidUri| {
            crabllm_core::Error::Internal(format!("bad url: {e}"))
        })?;
        let host = parsed
            .host()
            .ok_or_else(|| crabllm_core::Error::Internal("url has no host".to_string()))?;
        let path = parsed.path();
        let query = parsed.query().unwrap_or("");

        let now = now_utc();
        let date_stamp = &now[..8];
        let amz_date = &now;

        let content_hash = hex_sha256(body);
        let credential_scope = format!("{date_stamp}/{region}/{SERVICE}/aws4_request");

        let canonical_headers = format!(
            "content-type:application/json\nhost:{host}\nx-amz-content-sha256:{content_hash}\nx-amz-date:{amz_date}\n"
        );
        let signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date";

        let canonical_request = format!(
            "{method}\n{path}\n{query}\n{canonical_headers}\n{signed_headers}\n{content_hash}"
        );
        let canonical_request_hash = hex_sha256(canonical_request.as_bytes());

        let string_to_sign =
            format!("AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{canonical_request_hash}");

        let signing_key = derive_signing_key(secret_key, date_stamp, region);
        let signature = hex_hmac_sha256(&signing_key, string_to_sign.as_bytes());

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
        );

        Ok(vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("host".to_string(), host.to_string()),
            ("x-amz-date".to_string(), amz_date.to_string()),
            ("x-amz-content-sha256".to_string(), content_hash),
            ("authorization".to_string(), authorization),
        ])
    }

    fn hex_sha256(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex_encode(&hasher.finalize())
    }

    fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
        let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
        mac.update(data);
        mac.finalize().into_bytes().to_vec()
    }

    fn hex_hmac_sha256(key: &[u8], data: &[u8]) -> String {
        hex_encode(&hmac_sha256(key, data))
    }

    fn derive_signing_key(secret_key: &str, date_stamp: &str, region: &str) -> Vec<u8> {
        let k_date = hmac_sha256(
            format!("AWS4{secret_key}").as_bytes(),
            date_stamp.as_bytes(),
        );
        let k_region = hmac_sha256(&k_date, region.as_bytes());
        let k_service = hmac_sha256(&k_region, SERVICE.as_bytes());
        hmac_sha256(&k_service, b"aws4_request")
    }

    fn hex_encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            write!(s, "{b:02x}").unwrap();
        }
        s
    }

    fn now_utc() -> String {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_secs();

        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        let (year, month, day) = civil_from_days(days as i64);

        format!("{year:04}{month:02}{day:02}T{hours:02}{minutes:02}{seconds:02}Z")
    }

    fn civil_from_days(days: i64) -> (i32, u32, u32) {
        let z = days + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = (z - era * 146097) as u32;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        (y as i32, m, d)
    }
}

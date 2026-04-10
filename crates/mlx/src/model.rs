//! `MlxModel` — one loaded MLX model wrapped in the `Provider` trait.
//!
//! This is the Phase 6 primitive: take an OpenAI-shape
//! `ChatCompletionRequest`, serialize its messages + tools to JSON,
//! hand it to the Phase 3 [`Session`] via `spawn_blocking`, reassemble
//! the output into a `ChatCompletionResponse` / `ChatCompletionChunk`
//! stream.
//!
//! Cloning a `MlxModel` is cheap (`Arc<Session>` underneath) so
//! callers who want a pool hold one `MlxModel` per loaded weight and
//! clone it into each [`crate::pool::MlxPool`] entry.

use crate::{
    download,
    session::{GenerateOptions, GenerateRequest, Session},
};
use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, Delta, Error, FinishReason, FunctionCall, FunctionCallDelta, Message, Provider,
    Role, ToolCall, ToolCallDelta, ToolType, Usage,
};
use futures::{channel::mpsc, stream::StreamExt};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

/// One loaded MLX model. Cheap to `Clone`.
#[derive(Clone)]
pub struct MlxModel {
    /// The model name the caller passed to [`MlxModel::new`] — used
    /// verbatim in response `model` fields so downstream UIs show the
    /// same label the request carried.
    name: String,
    model_dir: PathBuf,
    session: Arc<Session>,
}

impl MlxModel {
    /// Load a model by path or HuggingFace repo id.
    ///
    /// If `model_id` is an existing directory, it is used as-is.
    /// Otherwise it is treated as a HuggingFace repo like
    /// `mlx-community/Qwen3.5-0.8B-Instruct-4bit` and fetched via
    /// [`crate::download::download_model`] into the default cache
    /// (`~/.crabtalk/cache/mlx/`). Cached models are reused without
    /// re-downloading.
    pub async fn new(model_id: impl Into<String>) -> Result<Self, Error> {
        let name = model_id.into();
        let model_dir = Self::resolve_path(&name).await?;
        let dir_for_load = model_dir.clone();
        let session = tokio::task::spawn_blocking(move || Session::new(&dir_for_load))
            .await
            .map_err(|e| Error::Internal(format!("mlx: load task panicked: {e}")))??;
        Ok(Self {
            name,
            model_dir,
            session: Arc::new(session),
        })
    }

    /// Build an `MlxModel` directly from an already-loaded
    /// [`Session`]. Useful for callers that manage downloads or
    /// session lifecycle themselves.
    pub fn from_session(name: impl Into<String>, model_dir: PathBuf, session: Session) -> Self {
        Self {
            name: name.into(),
            model_dir,
            session: Arc::new(session),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    async fn resolve_path(model_id: &str) -> Result<PathBuf, Error> {
        let as_path = Path::new(model_id);
        if as_path.exists() && as_path.is_dir() {
            return Ok(as_path.to_path_buf());
        }

        let cache_dir = download::default_cache_dir()?;
        if let Some(cached) = download::cached_model_path(model_id, &cache_dir) {
            return Ok(cached);
        }

        let repo = model_id.to_string();
        tokio::task::spawn_blocking(move || {
            download::download_model(&repo, &cache_dir, &|downloaded, total| {
                if let Some(total) = total {
                    tracing::debug!(downloaded = downloaded, total = total, "mlx: downloading");
                }
            })
        })
        .await
        .map_err(|e| Error::Internal(format!("mlx: download task panicked: {e}")))?
    }
}

impl Provider for MlxModel {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let messages_json = serialize_messages(&request.messages)?;
        let tools_json = serialize_tools(request)?;
        let options = request_options(request);
        let model_name = self.name.clone();
        let session = Arc::clone(&self.session);

        let output = tokio::task::spawn_blocking(move || {
            let req = GenerateRequest {
                messages_json: &messages_json,
                tools_json: tools_json.as_deref(),
                options,
                cancel_flag: None,
            };
            session.generate(&req)
        })
        .await
        .map_err(|e| Error::Internal(format!("mlx: generate task panicked: {e}")))??;

        let tool_calls = parse_tool_calls(output.tool_calls_json.as_deref())?;
        let finish_reason = if tool_calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        };

        Ok(ChatCompletionResponse {
            id: new_completion_id(),
            object: "chat.completion".to_string(),
            created: unix_seconds_now(),
            model: model_name,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: Some(serde_json::Value::String(output.text)),
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                    extra: serde_json::Map::new(),
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: Some(Usage {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
                total_tokens: output.prompt_tokens + output.completion_tokens,
                completion_tokens_details: None,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
            }),
            system_fingerprint: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let messages_json = serialize_messages(&request.messages)?;
        let tools_json = serialize_tools(request)?;
        let options = request_options(request);
        let model_name = self.name.clone();
        let session = Arc::clone(&self.session);
        let id = new_completion_id();
        let created = unix_seconds_now();

        let (tx, rx) = mpsc::unbounded::<Result<ChatCompletionChunk, Error>>();

        // First chunk carries only the role so the client knows the
        // assistant is responding. Matches OpenAI's SSE shape.
        let role_chunk = role_chunk(&id, created, &model_name);
        let _ = tx.unbounded_send(Ok(role_chunk));

        let id_for_task = id.clone();
        let model_for_task = model_name.clone();
        let tx_for_task = tx.clone();

        tokio::task::spawn_blocking(move || {
            let req = GenerateRequest {
                messages_json: &messages_json,
                tools_json: tools_json.as_deref(),
                options,
                cancel_flag: None,
            };
            let result = session.generate_stream(&req, |chunk| {
                let delta = content_chunk(&id_for_task, created, &model_for_task, chunk);
                // Returns Err if the receiver has been dropped; treat
                // that as "stop" so Swift releases the GPU.
                tx_for_task.unbounded_send(Ok(delta)).is_err()
            });

            match result {
                Ok(out) => {
                    let tool_calls = match parse_tool_calls(out.tool_calls_json.as_deref()) {
                        Ok(tc) => tc,
                        Err(e) => {
                            let _ = tx_for_task.unbounded_send(Err(e));
                            return;
                        }
                    };
                    let final_chunk = final_chunk(
                        &id_for_task,
                        created,
                        &model_for_task,
                        &tool_calls,
                        out.prompt_tokens,
                        out.completion_tokens,
                    );
                    let _ = tx_for_task.unbounded_send(Ok(final_chunk));
                }
                Err(e) => {
                    let _ = tx_for_task.unbounded_send(Err(e));
                }
            }
            // Dropping `tx_for_task` closes the channel; the consumer
            // sees end-of-stream after the final chunk.
        });

        // Drop the original sender so the stream terminates when the
        // spawned task drops its clone.
        drop(tx);

        Ok(rx.boxed())
    }
}

// ---------- request → FFI translation ----------

fn request_options(request: &ChatCompletionRequest) -> GenerateOptions {
    GenerateOptions {
        seed: request.seed.unwrap_or(0),
        max_tokens: request.max_tokens.unwrap_or(0),
        // 0.0 is a sentinel for "Swift default" — the Swift side
        // treats any non-positive value as unset. Any user who
        // genuinely wants temperature=0 gets a tiny epsilon
        // rounded-away in Swift, which is also "greedy-ish".
        temperature: request.temperature.map(|t| t as f32).unwrap_or(0.0),
        top_p: request.top_p.map(|t| t as f32).unwrap_or(0.0),
    }
}

fn serialize_messages(messages: &[Message]) -> Result<String, Error> {
    #[derive(serde::Serialize)]
    struct Out<'a> {
        role: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
    }

    let mut out: Vec<Out<'_>> = Vec::with_capacity(messages.len());
    for msg in messages {
        out.push(Out {
            role: msg.role.as_str(),
            // We flatten multimodal content to its text view: Phase 5
            // only supports text LLMs. Once VLM lands the Swift side
            // will parse richer JSON and this helper will grow
            // image_url / input_audio handling.
            content: msg.content_str().map(|s| s.to_string()).or_else(|| {
                msg.content
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .map(String::from)
            }),
        });
    }
    serde_json::to_string(&out)
        .map_err(|e| Error::Internal(format!("mlx: serialize messages: {e}")))
}

fn serialize_tools(request: &ChatCompletionRequest) -> Result<Option<String>, Error> {
    let Some(tools) = request.tools.as_ref() else {
        return Ok(None);
    };
    if tools.is_empty() {
        return Ok(None);
    }
    // The OpenAI-shape Tool struct in crabllm_core matches what
    // swift-transformers' ToolSpec expects once flattened to a JSON
    // dict. We re-emit it as an array so Swift can iterate.
    serde_json::to_string(tools)
        .map(Some)
        .map_err(|e| Error::Internal(format!("mlx: serialize tools: {e}")))
}

// ---------- response assembly ----------

fn new_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn role_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some(Role::Assistant),
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
    }
}

fn content_chunk(id: &str, created: u64, model: &str, text: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(text.to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
    }
}

fn final_chunk(
    id: &str,
    created: u64,
    model: &str,
    tool_calls: &[ToolCall],
    prompt_tokens: u32,
    completion_tokens: u32,
) -> ChatCompletionChunk {
    let finish_reason = if tool_calls.is_empty() {
        FinishReason::Stop
    } else {
        FinishReason::ToolCalls
    };
    let tool_call_deltas: Option<Vec<ToolCallDelta>> = if tool_calls.is_empty() {
        None
    } else {
        Some(
            tool_calls
                .iter()
                .enumerate()
                .map(|(idx, tc)| ToolCallDelta {
                    index: idx as u32,
                    id: Some(tc.id.clone()),
                    kind: Some(tc.kind),
                    function: Some(FunctionCallDelta {
                        name: Some(tc.function.name.clone()),
                        arguments: Some(tc.function.arguments.clone()),
                    }),
                })
                .collect(),
        )
    };
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                tool_calls: tool_call_deltas,
                reasoning_content: None,
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
        }),
        system_fingerprint: None,
    }
}

/// Parse the `tool_calls_json` field Swift returns. The Swift side
/// encodes `ToolCall` structs via `JSONEncoder`, which produces an
/// array of `{function: {name, arguments}}` objects. We wrap them in
/// our OpenAI-shape `ToolCall` with a synthesized id + `"function"`
/// type.
fn parse_tool_calls(json: Option<&str>) -> Result<Vec<ToolCall>, Error> {
    let Some(json) = json else {
        return Ok(Vec::new());
    };
    #[derive(serde::Deserialize)]
    struct RawToolCall {
        function: RawFunction,
    }
    #[derive(serde::Deserialize)]
    struct RawFunction {
        name: String,
        arguments: serde_json::Value,
    }

    let raw: Vec<RawToolCall> = serde_json::from_str(json)
        .map_err(|e| Error::Internal(format!("mlx: parse tool_calls_json: {e}")))?;
    Ok(raw
        .into_iter()
        .enumerate()
        .map(|(idx, call)| ToolCall {
            index: Some(idx as u32),
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            kind: ToolType::Function,
            function: FunctionCall {
                name: call.function.name,
                // OpenAI expects `arguments` as a JSON string, not an
                // object. Re-serialize whatever Swift gave us.
                arguments: call.function.arguments.to_string(),
            },
        })
        .collect())
}

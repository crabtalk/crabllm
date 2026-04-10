//! `MlxProvider` — `Provider`-trait front door backed by a Swift pool.
//!
//! The multi-model cache, idle eviction, and model lifecycle all live
//! in Swift (see `mlx/Sources/CrabllmMlx/Pool.swift`). This module is
//! a thin Rust shim that:
//!   1. Resolves the model name to a local directory (downloading from
//!      HuggingFace if needed).
//!   2. Calls the Swift pool FFI via `spawn_blocking`.
//!   3. Reassembles the results into OpenAI-shape types.
//!
//! Cloning is cheap: `Arc<MlxPool>` underneath.

use crate::{
    download,
    pool::MlxPool,
    session::{GenerateOptions, GenerateRequest},
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

#[derive(Clone)]
pub struct MlxProvider {
    pool: Arc<MlxPool>,
}

impl std::fmt::Debug for MlxProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxProvider").finish_non_exhaustive()
    }
}

impl MlxProvider {
    pub fn new(pool: Arc<MlxPool>) -> Self {
        Self { pool }
    }

    /// Resolve a model name to a local directory path. If it's already
    /// a directory, use it. Otherwise treat it as a HuggingFace repo id
    /// and download via `hf-hub` (respects `$HF_TOKEN`, `$HF_ENDPOINT`).
    async fn resolve_model_dir(&self, model_id: &str) -> Result<PathBuf, Error> {
        let as_path = Path::new(model_id);
        if as_path.exists() && as_path.is_dir() {
            return Ok(as_path.to_path_buf());
        }

        if let Some(cached) = download::cached_model_path(model_id) {
            return Ok(cached);
        }

        let repo = model_id.to_string();
        tokio::task::spawn_blocking(move || download::download_model(&repo))
            .await
            .map_err(|e| Error::Internal(format!("mlx: download task panicked: {e}")))?
    }
}

impl Provider for MlxProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let model_dir = self.resolve_model_dir(&request.model).await?;
        let model_dir_str = model_dir.to_string_lossy().to_string();
        let messages_json = serialize_messages(&request.messages)?;
        let tools_json = serialize_tools(request)?;
        let options = request_options(request);
        let model_name = request.model.clone();
        let pool = Arc::clone(&self.pool);

        let output = tokio::task::spawn_blocking(move || {
            let req = GenerateRequest {
                messages_json: &messages_json,
                tools_json: tools_json.as_deref(),
                options,
                cancel_flag: None,
            };
            pool.generate(&model_dir_str, &req)
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
        let model_dir = self.resolve_model_dir(&request.model).await?;
        let model_dir_str = model_dir.to_string_lossy().to_string();
        let messages_json = serialize_messages(&request.messages)?;
        let tools_json = serialize_tools(request)?;
        let options = request_options(request);
        let model_name = request.model.clone();
        let pool = Arc::clone(&self.pool);
        let id = new_completion_id();
        let created = unix_seconds_now();

        let (tx, rx) = mpsc::unbounded::<Result<ChatCompletionChunk, Error>>();

        let role_chunk = make_role_chunk(&id, created, &model_name);
        let _ = tx.unbounded_send(Ok(role_chunk));

        let id_c = id.clone();
        let model_c = model_name.clone();
        let tx_c = tx.clone();

        tokio::task::spawn_blocking(move || {
            let req = GenerateRequest {
                messages_json: &messages_json,
                tools_json: tools_json.as_deref(),
                options,
                cancel_flag: None,
            };
            let result = pool.generate_stream(&model_dir_str, &req, |chunk| {
                let delta = make_content_chunk(&id_c, created, &model_c, chunk);
                tx_c.unbounded_send(Ok(delta)).is_err()
            });

            match result {
                Ok(out) => {
                    let tool_calls = match parse_tool_calls(out.tool_calls_json.as_deref()) {
                        Ok(tc) => tc,
                        Err(e) => {
                            let _ = tx_c.unbounded_send(Err(e));
                            return;
                        }
                    };
                    let final_c = make_final_chunk(
                        &id_c,
                        created,
                        &model_c,
                        &tool_calls,
                        out.prompt_tokens,
                        out.completion_tokens,
                    );
                    let _ = tx_c.unbounded_send(Ok(final_c));
                }
                Err(e) => {
                    let _ = tx_c.unbounded_send(Err(e));
                }
            }
        });

        drop(tx);
        Ok(rx.boxed())
    }
}

// ---------- shared helpers (copied from model.rs, which is being deleted) ----------

fn request_options(request: &ChatCompletionRequest) -> GenerateOptions {
    GenerateOptions {
        max_tokens: request.max_tokens.unwrap_or(0),
        temperature: request.temperature.map(|t| t as f32).unwrap_or(0.0),
        top_p: request.top_p.map(|t| t as f32).unwrap_or(0.0),
    }
}

fn serialize_messages(messages: &[crabllm_core::Message]) -> Result<String, Error> {
    // Serialize the full Message structs — they include role, content,
    // tool_calls, tool_call_id, name, and reasoning_content. The Swift
    // side parses what it needs via JSONSerialization.
    serde_json::to_string(messages)
        .map_err(|e| Error::Internal(format!("mlx: serialize messages: {e}")))
}

fn serialize_tools(request: &ChatCompletionRequest) -> Result<Option<String>, Error> {
    let Some(tools) = request.tools.as_ref() else {
        return Ok(None);
    };
    if tools.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(tools)
        .map(Some)
        .map_err(|e| Error::Internal(format!("mlx: serialize tools: {e}")))
}

fn new_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn make_role_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
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

fn make_content_chunk(id: &str, created: u64, model: &str, text: &str) -> ChatCompletionChunk {
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

fn make_final_chunk(
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
                arguments: call.function.arguments.to_string(),
            },
        })
        .collect())
}

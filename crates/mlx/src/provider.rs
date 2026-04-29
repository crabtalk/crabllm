//! `MlxProvider` — `Provider`-trait front door backed by a Swift pool.
//!
//! The multi-model cache, idle eviction, and model lifecycle all live
//! in Swift (see `mlx/Sources/CrabllmMlx/Pool.swift`). This module is
//! a thin Rust shim that:
//!   1. Resolves the model name to a cached local directory (errors
//!      if not downloaded — use `download::download_model` first).
//!   2. Calls the Swift pool FFI via `spawn_blocking`.
//!   3. Reassembles the results into OpenAI-shape types.
//!
//! Cloning is cheap: `Arc<MlxPool>` underneath.

use crate::{
    download,
    pool::{LoadedModel, MlxPool},
    session::{GenerateOptions, GenerateRequest},
};
use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, Delta, Error, FinishReason, FunctionCall, FunctionCallDelta, Message, Provider,
    Role, ToolCall, ToolCallDelta, ToolType, Usage,
};
use futures::{channel::mpsc, stream::StreamExt};
use std::{
    fs,
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

    /// Resolve a model alias to a HF repo ID via the build-time
    /// registry, passing through as-is if not found.
    fn resolve_repo_id<'a>(&'a self, model_id: &'a str) -> &'a str {
        crate::registry::resolve(model_id).unwrap_or(model_id)
    }

    /// Snapshot the pool's loaded-model inventory. See
    /// [`MlxPool::loaded_models`] for the full contract.
    pub fn loaded_models(&self) -> Result<Vec<LoadedModel>, Error> {
        self.pool.loaded_models()
    }

    /// Resolve a model name to a local directory path.
    ///
    /// Accepts: local directory path, full HF repo id, or a registry
    /// alias. Does NOT auto-download — returns an error if the model
    /// is not cached. Use [`download::download_model`] to download
    /// explicitly before calling the provider.
    fn resolve_model_dir(&self, model_id: &str) -> Result<PathBuf, Error> {
        self.lookup_cached_model_dir(model_id).ok_or_else(|| {
            let repo_id = self.resolve_repo_id(model_id);
            Error::Internal(format!(
                "mlx: model not downloaded: {repo_id}. \
                 Use download::download_model() first."
            ))
        })
    }

    fn lookup_cached_model_dir(&self, model_id: &str) -> Option<PathBuf> {
        let as_path = Path::new(model_id);
        if as_path.exists() && as_path.is_dir() {
            return Some(as_path.to_path_buf());
        }
        let repo_id = self.resolve_repo_id(model_id);
        download::cached_model_path(repo_id)
    }

    /// Unload a model from the pool.
    ///
    /// Accepts the same inputs as a generate call: a local directory
    /// path, a full HuggingFace repo id, or a registry alias.
    ///
    /// Returns `true` if the slot was loaded in the pool at the
    /// moment of the call and was actually evicted; `false` if the
    /// model was not cached on disk or was cached but not loaded in
    /// the pool. Downstream admin endpoints can map `false` to a
    /// 404 "not loaded" response and `true` to a 200 "evicted".
    ///
    /// Generations already in flight for this model continue
    /// uninterrupted: the Swift side holds a strong reference to the
    /// `ModelContainer` for the duration of the call, so dropping it
    /// from the pool's slot table only releases the pool's reference.
    /// The next request for this model reloads from disk.
    pub fn unload(&self, model_id: &str) -> bool {
        let Some(dir) = self.lookup_cached_model_dir(model_id) else {
            return false;
        };
        self.pool.evict(&dir.to_string_lossy())
    }

    /// Evict every loaded model and stop the idle monitor. The pool
    /// handle remains valid — subsequent requests will load models
    /// on demand but no new idle sweep will run.
    pub fn unload_all(&self) {
        self.pool.stop_all();
    }
}

impl Provider for MlxProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let model_dir = self.resolve_model_dir(&request.model)?;
        let model_dir_str = model_dir.to_string_lossy().to_string();
        let mut messages_json = serialize_messages(&request.messages)?;
        let mut tools_json = serialize_tools(request)?;
        if tools_json.is_some() {
            ensure_tool_calling_supported(&model_dir, &request.model)?;
        }
        let is_gemma = crate::gemma_patch::is_gemma_model(&model_dir);
        if is_gemma {
            messages_json = crate::gemma_patch::preprocess_messages_json(&messages_json)?;
            tools_json = tools_json
                .map(|t| crate::gemma_patch::preprocess_tools_json(&t))
                .transpose()?;
        }
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

        let mut tool_calls = parse_tool_calls(output.tool_calls_json.as_deref())?;
        // gemma-3/4 emit tool calls as raw text because mlx-swift-lm
        // 3.31.3 ships the wrong tags / model_type match. Recover them
        // here when the request asked for tools but nothing structured
        // came back. See `gemma_patch`.
        let mut text = output.text;
        if is_gemma && request.tools.is_some() && tool_calls.is_empty() {
            let (cleaned, extracted) = crate::gemma_patch::extract_gemma_tool_calls(&text);
            if !extracted.is_empty() {
                text = cleaned;
                tool_calls = extracted;
            }
        }
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
                    content: Some(serde_json::Value::String(text)),
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
        let model_dir = self.resolve_model_dir(&request.model)?;
        let model_dir_str = model_dir.to_string_lossy().to_string();
        let mut messages_json = serialize_messages(&request.messages)?;
        let mut tools_json = serialize_tools(request)?;
        if tools_json.is_some() {
            ensure_tool_calling_supported(&model_dir, &request.model)?;
        }
        let is_gemma = crate::gemma_patch::is_gemma_model(&model_dir);
        if is_gemma {
            messages_json = crate::gemma_patch::preprocess_messages_json(&messages_json)?;
            tools_json = tools_json
                .map(|t| crate::gemma_patch::preprocess_tools_json(&t))
                .transpose()?;
        }
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
        // Gemma's stale parser leaks tool calls as raw text, so when
        // tools are requested for a gemma model we buffer the whole
        // stream and recover them at end-of-stream (see `gemma_patch`).
        // Tool-using turns are short and atomic, so the loss of
        // token-level streaming is acceptable. Other models stream
        // chunks straight through.
        let buffer_for_tools = is_gemma && tools_json.is_some();

        tokio::task::spawn_blocking(move || {
            let req = GenerateRequest {
                messages_json: &messages_json,
                tools_json: tools_json.as_deref(),
                options,
                cancel_flag: None,
            };
            let mut buffered = String::new();
            let result = pool.generate_stream(&model_dir_str, &req, |chunk| {
                if buffer_for_tools {
                    buffered.push_str(chunk);
                    return false;
                }
                let delta = make_content_chunk(&id_c, created, &model_c, chunk);
                tx_c.unbounded_send(Ok(delta)).is_err()
            });

            match result {
                Ok(out) => {
                    let mut tool_calls = match parse_tool_calls(out.tool_calls_json.as_deref()) {
                        Ok(tc) => tc,
                        Err(e) => {
                            let _ = tx_c.unbounded_send(Err(e));
                            return;
                        }
                    };
                    if buffer_for_tools {
                        let mut text = std::mem::take(&mut buffered);
                        if tool_calls.is_empty() {
                            let (cleaned, extracted) =
                                crate::gemma_patch::extract_gemma_tool_calls(&text);
                            if !extracted.is_empty() {
                                text = cleaned;
                                tool_calls = extracted;
                            }
                        }
                        if !text.is_empty() {
                            let delta = make_content_chunk(&id_c, created, &model_c, &text);
                            let _ = tx_c.unbounded_send(Ok(delta));
                        }
                    }
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

/// Reject tool-augmented requests against models whose chat template
/// has no tool slot. Without this guard, mlx-swift-lm renders the
/// prompt as if `tools` were never passed, the model immediately
/// emits EOS, and the caller sees a silent zero-token "success".
///
/// Detection reads the on-disk chat template (either a standalone
/// `chat_template.json` or the `chat_template` field inside
/// `tokenizer_config.json`) and looks for a `tools` reference. The
/// check is conservative — if we can't read the file we let the
/// request through rather than block legitimate work.
fn ensure_tool_calling_supported(model_dir: &Path, model_name: &str) -> Result<(), Error> {
    let Some(template) = read_chat_template(model_dir) else {
        return Ok(());
    };
    if chat_template_supports_tools(&template) {
        return Ok(());
    }
    Err(Error::Internal(format!(
        "model '{model_name}' does not support tool calling — \
         retry without tools or pick a tool-capable model"
    )))
}

/// Load the chat template as a single JSON value (string or array of
/// `{name, template}` entries). Returns `None` when no template file
/// is present or readable — let the request fall through.
///
/// Looks in three places, in HuggingFace's order of precedence:
///   1. `chat_template.jinja` — modern raw-Jinja sidecar (gemma-3+,
///      llama-3.1+, qwen-3, etc. all ship this way).
///   2. `chat_template.json` — `{ "chat_template": <string|array> }`
///      legacy sidecar.
///   3. `tokenizer_config.json` — embedded `chat_template` field used
///      by older repos.
fn read_chat_template(model_dir: &Path) -> Option<serde_json::Value> {
    if let Ok(s) = fs::read_to_string(model_dir.join("chat_template.jinja")) {
        return Some(serde_json::Value::String(s));
    }
    if let Ok(bytes) = fs::read(model_dir.join("chat_template.json"))
        && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
    {
        if let Some(t) = v.get("chat_template").cloned() {
            return Some(t);
        }
        return Some(v);
    }
    let bytes = fs::read(model_dir.join("tokenizer_config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    v.get("chat_template").cloned()
}

/// Heuristic mirror of HuggingFace's "supports tool calling" check:
/// a template advertises tool support if it is the array form with a
/// `tool_use` entry, or a literal Jinja string that references the
/// `tools` variable.
fn chat_template_supports_tools(template: &serde_json::Value) -> bool {
    match template {
        serde_json::Value::String(s) => s.contains("tools"),
        serde_json::Value::Array(items) => items.iter().any(|item| {
            item.get("name").and_then(|n| n.as_str()) == Some("tool_use")
                || item
                    .get("template")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t.contains("tools"))
        }),
        _ => false,
    }
}

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

fn random_hex() -> String {
    use rand::Rng;
    let bytes: [u8; 12] = rand::rng().random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn new_completion_id() -> String {
    format!("chatcmpl-{}", random_hex())
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
            id: format!("call_{}", random_hex()),
            kind: ToolType::Function,
            function: FunctionCall {
                name: call.function.name,
                arguments: call.function.arguments.to_string(),
            },
        })
        .collect())
}

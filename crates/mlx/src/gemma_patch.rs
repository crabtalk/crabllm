//! Gemma-3/4 chat-template + tool-calling patches.
//!
//! Everything in this module exists to compensate for bugs in
//! mlx-swift-lm 3.31.3 and quirks in gemma-3/4's bundled
//! `chat_template.jinja`. All entry points are intended to be gated
//! by [`is_gemma_model`] in `provider.rs` so other model families
//! never see these transforms. When the upstream issues tracked
//! below are resolved, this module and its call sites should be
//! removed in a single revert.
//!
//! ## Tracked issues
//!
//! - **mlx-swift-lm #215** —
//!   <https://github.com/ml-explore/mlx-swift-lm/pull/215>.
//!   `GemmaFunctionParser` looks for the wrong delimiters
//!   (`<start_function_call>` instead of `<|tool_call>`) and
//!   `ToolCallFormat.infer` only matches `model_type == "gemma"`,
//!   so `gemma3` / `gemma4` fall through to the JSON parser and
//!   tool calls leak as raw text. Drives both
//!   [`extract_gemma_tool_calls`] (parse direction) and
//!   [`preprocess_messages_json`] (emit direction —
//!   `Chat.Message.assistant` has no `tool_calls` field, so
//!   replayed assistant turns must round-trip through gemma-format
//!   text in `content`).
//!
//! - **Gemma chat-template `| upper` on union `type`** — the
//!   bundled `chat_template.jinja` applies `{{ type | upper }}`
//!   directly to a tool parameter's `type` field. Jinja's `upper`
//!   requires a string, so the standard nullable shape
//!   `["string", "null"]` crashes the renderer.
//!   [`preprocess_tools_json`] flattens these unions before the
//!   template ever sees them. No upstream tracking issue yet — the
//!   fix lives in gemma's template repo, not mlx-swift-lm.
//!
//! - **mlx-swift-lm Jinja value coercer rejects `NSNull`** —
//!   `JSONSerialization` decodes JSON `null` to `NSNull`, which
//!   bridges to `Optional<Any>` when the coercer reaches it; it
//!   fails with `"Cannot convert value of type Optional<Any> to
//!   Jinja Value"`. Real schemas commonly carry `"default": null`,
//!   `"description": null`, etc. [`preprocess_tools_json`] strips
//!   every `null` leaf as part of the same walk. No upstream
//!   tracking issue yet.

use crabllm_core::{Error, FunctionCall, ToolCall, ToolType};
use rand::Rng;
use serde_json::{Map, Value};
use std::{fs, path::Path};

const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_ESCAPE: &str = "<|\"|>";

/// True if `model_dir/config.json` advertises a gemma family
/// (`model_type` starting with `gemma`). Conservative: any I/O,
/// parse, or shape failure returns `false` — misclassifying a
/// non-gemma model as gemma would corrupt its prompts.
pub(crate) fn is_gemma_model(model_dir: &Path) -> bool {
    let bytes = match fs::read(model_dir.join("config.json")) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let v: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return false,
    };
    v.get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|s| s.starts_with("gemma"))
}

// ---------- tool schema preprocessing ----------

/// Flatten union `type` fields and strip JSON `null` leaves from a
/// tools-array JSON document. Returns the rewritten JSON. Both
/// transforms run in a single recursive walk.
///
/// Known limitation: a union of two non-null types
/// (`["string", "integer"]`) is truncated to its first member.
/// Rare in real-world schemas; worth tracking separately.
pub(crate) fn preprocess_tools_json(tools_json: &str) -> Result<String, Error> {
    let mut v: Value = serde_json::from_str(tools_json)
        .map_err(|e| Error::Internal(format!("mlx: tools_json parse: {e}")))?;
    walk_schema(&mut v);
    serde_json::to_string(&v)
        .map_err(|e| Error::Internal(format!("mlx: tools_json reserialize: {e}")))
}

fn walk_schema(v: &mut Value) {
    match v {
        Value::Object(map) => {
            // type: ["string", "null"] → type: "string", nullable: true
            if let Some(Value::Array(arr)) = map.get("type").cloned() {
                let strs: Vec<&str> = arr.iter().filter_map(Value::as_str).collect();
                let has_null = strs.iter().any(|s| *s == "null");
                if let Some(chosen) = strs.iter().find(|s| **s != "null") {
                    map.insert("type".to_string(), Value::String((*chosen).to_string()));
                    if has_null && !map.contains_key("nullable") {
                        map.insert("nullable".to_string(), Value::Bool(true));
                    }
                } else if has_null {
                    map.insert("type".to_string(), Value::String("null".to_string()));
                }
            }
            // Strip null leaves and recurse into children.
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                match map.get_mut(&key) {
                    Some(Value::Null) => {
                        map.remove(&key);
                    }
                    Some(child) => walk_schema(child),
                    None => {}
                }
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                walk_schema(item);
            }
        }
        _ => {}
    }
}

// ---------- messages preprocessing ----------

/// Walk an OpenAI-shape messages array. For every assistant message
/// with `tool_calls`, render those calls into gemma-format text and
/// append them to `content`, then drop the `tool_calls` field.
/// `Chat.Message.assistant` (mlx-swift-lm) has no `tool_calls`
/// parameter, so prior tool-using turns can only re-enter the
/// prompt via `content`.
pub(crate) fn preprocess_messages_json(messages_json: &str) -> Result<String, Error> {
    let mut v: Value = serde_json::from_str(messages_json)
        .map_err(|e| Error::Internal(format!("mlx: messages_json parse: {e}")))?;
    let Some(arr) = v.as_array_mut() else {
        return Err(Error::Internal(
            "mlx: messages_json is not a JSON array".to_string(),
        ));
    };
    for msg in arr.iter_mut() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if obj.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(Value::Array(calls)) = obj.remove("tool_calls") else {
            continue;
        };
        if calls.is_empty() {
            continue;
        }
        let suffix = render_gemma_tool_calls(&calls);
        let existing = obj.remove("content").unwrap_or(Value::Null);
        obj.insert("content".to_string(), append_text_content(existing, suffix));
    }
    serde_json::to_string(&v)
        .map_err(|e| Error::Internal(format!("mlx: messages_json reserialize: {e}")))
}

fn append_text_content(content: Value, suffix: String) -> Value {
    match content {
        Value::String(s) => Value::String(s + &suffix),
        Value::Null => Value::String(suffix),
        Value::Array(mut parts) => {
            parts.push(serde_json::json!({"type": "text", "text": suffix}));
            Value::Array(parts)
        }
        other => other,
    }
}

fn render_gemma_tool_calls(calls: &[Value]) -> String {
    let mut out = String::new();
    for call in calls {
        let Some(function) = call.get("function") else {
            continue;
        };
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            continue;
        };
        out.push_str(TOOL_CALL_START);
        out.push_str("call:");
        out.push_str(name);
        out.push('{');
        let args_str = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        let args: Map<String, Value> = serde_json::from_str(args_str)
            .ok()
            .and_then(|v: Value| match v {
                Value::Object(m) => Some(m),
                _ => None,
            })
            .unwrap_or_default();
        // Sort keys for stable output — matches gemma's
        // template-side `dictsort` filter and means re-renders are
        // byte-identical across runs.
        let mut keys: Vec<&String> = args.keys().collect();
        keys.sort();
        for (i, key) in keys.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(key);
            out.push(':');
            out.push_str(&format_gemma_arg_value(&args[*key]));
        }
        out.push('}');
        out.push_str(TOOL_CALL_END);
    }
    out
}

fn format_gemma_arg_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("{STRING_ESCAPE}{s}{STRING_ESCAPE}"),
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

// ---------- output extraction ----------

/// Extract every gemma-format tool call from `text`. Returns the
/// text with all `<|tool_call>...<tool_call|>` blocks removed, plus
/// the parsed calls. Until mlx-swift-lm #215 lands, gemma-3/4 reach
/// us shaped exactly as the template emits them
/// (chat_template.jinja:192-203):
///
///   `<|tool_call>call:NAME{key:value,key:<|"|>string<|"|>,...}<tool_call|>`
pub(crate) fn extract_gemma_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(TOOL_CALL_START) {
        clean.push_str(&rest[..start]);
        let after_start = &rest[start + TOOL_CALL_START.len()..];
        let Some(end) = after_start.find(TOOL_CALL_END) else {
            // Unterminated marker — preserve original text rather
            // than silently swallow it.
            clean.push_str(&rest[start..]);
            return (clean, calls);
        };
        let body = &after_start[..end];
        if let Some(call) = parse_call_body(body, calls.len()) {
            calls.push(call);
        }
        rest = &after_start[end + TOOL_CALL_END.len()..];
    }
    clean.push_str(rest);
    (clean, calls)
}

fn parse_call_body(body: &str, index: usize) -> Option<ToolCall> {
    let body = body.trim().strip_prefix("call:")?;
    let brace = body.find('{')?;
    let name = body[..brace].trim();
    if name.is_empty() {
        return None;
    }
    let inner = body[brace + 1..].trim_end();
    let inner = inner.strip_suffix('}')?;
    let args = parse_args(inner);
    Some(ToolCall {
        index: Some(index as u32),
        id: format!("call_{}", random_hex()),
        kind: ToolType::Function,
        function: FunctionCall {
            name: name.to_string(),
            arguments: Value::Object(args).to_string(),
        },
    })
}

fn parse_args(body: &str) -> Map<String, Value> {
    let mut out = Map::new();
    let mut rest = body.trim();
    while !rest.is_empty() {
        let Some(colon) = rest.find(':') else { break };
        let key = rest[..colon].trim().to_string();
        let after_colon = rest[colon + 1..].trim_start();
        let (value, consumed) = parse_value(after_colon);
        if !key.is_empty() {
            out.insert(key, value);
        }
        rest = after_colon[consumed..].trim_start();
        match rest.strip_prefix(',') {
            Some(after_comma) => rest = after_comma.trim_start(),
            None => break,
        }
    }
    out
}

fn parse_value(s: &str) -> (Value, usize) {
    if let Some(rest) = s.strip_prefix(STRING_ESCAPE) {
        return match rest.find(STRING_ESCAPE) {
            Some(close) => (
                Value::String(rest[..close].to_string()),
                STRING_ESCAPE.len() + close + STRING_ESCAPE.len(),
            ),
            None => (
                Value::String(rest.to_string()),
                STRING_ESCAPE.len() + rest.len(),
            ),
        };
    }
    let i = s.find(',').unwrap_or(s.len());
    let raw = s[..i].trim_end();
    let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
    (value, i)
}

fn random_hex() -> String {
    let bytes: [u8; 12] = rand::rng().random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

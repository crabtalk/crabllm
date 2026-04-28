//! Post-process gemma-format tool calls from MLX text output.
//!
//! mlx-swift-lm 3.31.3's `GemmaFunctionParser` looks for the wrong
//! delimiters (`<start_function_call>` / `<end_function_call>`) and
//! `ToolCallFormat.infer` only matches `model_type == "gemma"` — gemma
//! 3/4 use `model_type == "gemma3"` / `"gemma4"` so they fall through
//! to the default JSON parser and the call leaks out as raw text.
//! Upstream PR #215 fixes both, but it's blocked on review.
//!
//! Until that lands, gemma-3/4 calls reach us shaped exactly as the
//! template emits them (chat_template.jinja:192-203):
//!
//!   <|tool_call>call:NAME{key:value,key:<|"|>string<|"|>,...}<tool_call|>
//!
//! We strip the wrappers, parse the body, and synthesize OpenAI-shape
//! `tool_calls`. Bare values are tried as JSON (numbers, bools, null)
//! first, then fall back to string.

use crabllm_core::{FunctionCall, ToolCall, ToolType};
use rand::Rng;
use serde_json::{Map, Value};

const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_ESCAPE: &str = "<|\"|>";

/// Extract every gemma-format tool call from `text`. Returns the text
/// with all `<|tool_call>...<tool_call|>` blocks removed, plus the
/// parsed calls.
pub(crate) fn extract_gemma_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(TOOL_CALL_START) {
        clean.push_str(&rest[..start]);
        let after_start = &rest[start + TOOL_CALL_START.len()..];
        let Some(end) = after_start.find(TOOL_CALL_END) else {
            // Unterminated marker — preserve original text rather than
            // silently swallow it.
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

/// Parse a `call:NAME{key:value,...}` body.
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

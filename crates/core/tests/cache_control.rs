//! Prompt-cache breakpoints across the wire formats.
//!
//! Anthropic caches everything *up to and including* a block marked with
//! `cache_control`, so the mark is a boundary between blocks rather than a
//! property of one. The IR carries it as [`ir::Content::CacheBreakpoint`];
//! Anthropic folds it onto the preceding block, and formats without explicit
//! caching drop it.
use crabllm_core::{ChatCompletionRequest, anthropic, ir};
use serde_json::{Value, json};

fn anthropic_request(body: Value) -> anthropic::Request {
    serde_json::from_value(body).expect("decode anthropic request")
}

fn openai_request(body: Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("decode openai request")
}

// ── Anthropic round trip ──

#[test]
fn a_marked_system_block_survives_the_ir() {
    let req = anthropic_request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "long stable preamble",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "volatile tail"}
        ],
        "messages": [{"role": "user", "content": "hi"}]
    }));

    let ir_req = ir::Request::from(req);

    // The mark becomes a breakpoint sitting after the block it ended on.
    let system = ir_req.system.as_ref().expect("system");
    assert!(matches!(system[0], ir::Content::Text(ref t) if t == "long stable preamble"));
    assert!(matches!(system[1], ir::Content::CacheBreakpoint));
    assert!(matches!(system[2], ir::Content::Text(ref t) if t == "volatile tail"));

    // And folds back onto that same block on the way out.
    let out = anthropic::Request::from(&ir_req);
    let wire = serde_json::to_value(&out).expect("serialize");
    assert_eq!(
        wire["system"],
        json!([
            {"type": "text", "text": "long stable preamble",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "volatile tail"}
        ])
    );
}

#[test]
fn an_unmarked_request_stays_unmarked() {
    let req = anthropic_request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 100,
        "system": [{"type": "text", "text": "preamble"}],
        "messages": [{"role": "user", "content": "hi"}]
    }));

    let ir_req = ir::Request::from(req);
    assert!(
        !ir_req
            .system
            .as_ref()
            .expect("system")
            .iter()
            .any(|c| matches!(c, ir::Content::CacheBreakpoint))
    );

    let wire = serde_json::to_value(anthropic::Request::from(&ir_req)).expect("serialize");
    assert!(
        !wire.to_string().contains("cache_control"),
        "unmarked request grew a cache_control: {wire}"
    );
}

#[test]
fn a_marked_tool_survives_the_ir() {
    let req = anthropic_request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "name": "get_weather",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"}
        }]
    }));

    let ir_req = ir::Request::from(req);
    assert!(
        ir_req.tools.as_ref().expect("tools")[0]
            .cache_control
            .is_some()
    );

    let wire = serde_json::to_value(anthropic::Request::from(&ir_req)).expect("serialize");
    assert_eq!(
        wire["tools"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

// ── OpenAI-shaped request aimed at an Anthropic backend ──

/// The LiteLLM extension: `cache_control` on an OpenAI-shaped message, so a
/// client can drive Anthropic prompt caching without speaking Anthropic.
#[test]
fn a_cache_mark_on_an_openai_message_reaches_anthropic() {
    let req = openai_request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "long stable preamble",
             "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "hi"}
        ]
    }));

    let out = anthropic::Request::from(&ir::Request::from(req));
    let wire = serde_json::to_value(&out).expect("serialize");

    assert_eq!(
        wire["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

#[test]
fn a_cache_mark_on_an_openai_tool_reaches_anthropic() {
    let req = openai_request(json!({
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
            "cache_control": {"type": "ephemeral"}
        }]
    }));

    let wire =
        serde_json::to_value(anthropic::Request::from(&ir::Request::from(req))).expect("serialize");

    assert_eq!(
        wire["tools"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

/// It is not an OpenAI field, so it must never go out on an OpenAI request —
/// unknown message keys are rejected there.
#[test]
fn a_cache_mark_never_reaches_an_openai_provider() {
    let req = openai_request(json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "preamble",
             "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "hi"}
        ]
    }));

    let ir_req = ir::Request::from(req);
    // The IR still knows about it...
    assert!(
        ir_req
            .system
            .as_ref()
            .expect("system")
            .iter()
            .any(|c| matches!(c, ir::Content::CacheBreakpoint))
    );

    // ...but it is dropped at the OpenAI wire.
    let wire = serde_json::to_value(ChatCompletionRequest::from(&ir_req)).expect("serialize");
    assert!(
        !wire.to_string().contains("cache_control"),
        "cache_control leaked to OpenAI: {wire}"
    );
}

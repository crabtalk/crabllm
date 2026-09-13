//! Thinking / reasoning-effort translation across the three wire formats.
//!
//! OpenAI spells it `reasoning_effort` (an enum), Gemini and pre-4.6 Anthropic
//! spell it as a token budget, and Anthropic 4.6 onward spells it as adaptive
//! thinking plus `output_config.effort`. The IR keeps whichever form arrived
//! and converts on the way out, so the same knob survives any route.
use crabllm_core::{
    ChatCompletionRequest, anthropic, gemini,
    ir::{self, Effort},
};
use serde_json::{Value, json};

fn ir_request(model: &str, thinking: Option<Effort>) -> ir::Request {
    ir::Request {
        model: model.into(),
        system: None,
        messages: vec![ir::Message::user("hi")],
        max_tokens: 32_000,
        temperature: None,
        top_p: None,
        stop: None,
        tools: None,
        tool_choice: None,
        thinking,
        stream: false,
    }
}

/// The table LiteLLM uses to bridge OpenAI's effort enum and the integer
/// budget Anthropic and Gemini take (`litellm/constants.py`).
#[test]
fn the_effort_budget_table_matches_litellm() {
    assert_eq!(Effort::None.budget_tokens(), 0);
    assert_eq!(Effort::Minimal.budget_tokens(), 128);
    assert_eq!(Effort::Low.budget_tokens(), 1024);
    assert_eq!(Effort::Medium.budget_tokens(), 2048);
    assert_eq!(Effort::High.budget_tokens(), 4096);
    assert_eq!(Effort::XHigh.budget_tokens(), 8192);
    assert_eq!(Effort::Max.budget_tokens(), 16384);
}

/// A budget buys the strongest level it covers, never one it can't afford.
#[test]
fn a_budget_rounds_down_to_a_level() {
    assert_eq!(Effort::from(8000u32).level(), Effort::High);
    assert_eq!(Effort::from(8192u32).level(), Effort::XHigh);
    assert_eq!(Effort::from(1024u32).level(), Effort::Low);
    assert_eq!(Effort::from(1023u32).level(), Effort::Minimal);
}

/// Gemini spells "no thinking" as a zero budget, which is a level, not a
/// budget of nothing.
#[test]
fn a_zero_budget_is_the_none_level() {
    assert_eq!(Effort::from(0u32), Effort::None);
}

/// Every level round-trips through its own budget.
#[test]
fn every_level_round_trips_through_its_budget() {
    for level in [
        Effort::Minimal,
        Effort::Low,
        Effort::Medium,
        Effort::High,
        Effort::XHigh,
        Effort::Max,
    ] {
        assert_eq!(Effort::from(level.budget_tokens()).level(), level);
        assert_eq!(level.to_string().parse::<Effort>(), Ok(level));
    }
}

// ── Anthropic client → OpenAI-shaped provider ──

#[test]
fn an_anthropic_budget_becomes_a_reasoning_effort() {
    let body = json!({
        "model": "o3",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 8000}
    });
    let req: anthropic::Request = serde_json::from_value(body).expect("decode anthropic request");
    let wire: Value = serde_json::to_value(ChatCompletionRequest::from(&ir::Request::from(req)))
        .expect("serialize openai request");

    // 8000 affords `high` (4096) but not `xhigh` (8192).
    assert_eq!(wire["reasoning_effort"], "high");
}

#[test]
fn thinking_disabled_stays_disabled() {
    let body = json!({
        "model": "o3",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"}
    });
    let req: anthropic::Request = serde_json::from_value(body).expect("decode anthropic request");
    let wire: Value = serde_json::to_value(ChatCompletionRequest::from(&ir::Request::from(req)))
        .expect("serialize openai request");

    assert_eq!(wire["reasoning_effort"], "none");
}

#[test]
fn no_thinking_sends_no_reasoning_effort() {
    let wire: Value = serde_json::to_value(ChatCompletionRequest::from(&ir_request("o3", None)))
        .expect("serialize openai request");

    assert!(wire.get("reasoning_effort").is_none(), "got {wire}");
}

// ── OpenAI client → Anthropic provider ──

#[test]
fn a_reasoning_effort_becomes_adaptive_thinking() {
    let body = json!({
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32000,
        "reasoning_effort": "high"
    });
    let req: ChatCompletionRequest = serde_json::from_value(body).expect("decode openai request");
    let out = anthropic::Request::from(&ir::Request::from(req));

    let thinking = out.thinking.expect("thinking config");
    assert_eq!(thinking.kind, "adaptive");
    assert_eq!(thinking.budget_tokens, None);
    assert_eq!(
        out.output_config.and_then(|c| c.effort),
        Some("high".to_string())
    );
}

/// A client's `budget_tokens` still routes — it lands on the level it affords
/// rather than on the integer field 4.7 rejects.
#[test]
fn a_client_budget_lands_on_an_effort_level() {
    let out =
        anthropic::Request::from(&ir_request("claude-opus-4-7", Some(Effort::Budget(50_000))));

    assert_eq!(out.thinking.expect("thinking").budget_tokens, None);
    assert_eq!(
        out.output_config.and_then(|c| c.effort),
        Some("max".to_string())
    );
}

/// An adaptive request arriving from a client is read back off `output_config`.
#[test]
fn an_adaptive_request_is_read_into_the_ir() {
    let body = json!({
        "model": "claude-opus-4-7",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"}
    });
    let req: anthropic::Request = serde_json::from_value(body).expect("decode anthropic request");

    assert_eq!(ir::Request::from(req).thinking, Some(Effort::XHigh));
}

/// Adaptive without an `output_config` is Anthropic's own default, not ours.
#[test]
fn adaptive_without_an_effort_is_high() {
    let body = json!({
        "model": "claude-opus-4-7",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"}
    });
    let req: anthropic::Request = serde_json::from_value(body).expect("decode anthropic request");

    assert_eq!(ir::Request::from(req).thinking, Some(Effort::High));
}

// ── Gemini ──

#[test]
fn a_gemini_thinking_budget_is_read_into_the_ir() {
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "maxOutputTokens": 32000,
            "thinkingConfig": {"thinkingBudget": 8192, "includeThoughts": true}
        }
    });
    let req: gemini::Request = serde_json::from_value(body).expect("decode gemini request");
    let ir_req = ir::Request::from(&req);

    let thinking = ir_req.thinking.expect("thinking");
    assert_eq!(thinking, Effort::Budget(8192));
    assert_eq!(thinking.level(), Effort::XHigh);
}

/// Gemini's `-1` means "let the model decide" — there is no budget to read a
/// level from, so it lands on the default rather than being treated as zero.
#[test]
fn a_dynamic_gemini_budget_lands_on_the_default_level() {
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": -1}}
    });
    let req: gemini::Request = serde_json::from_value(body).expect("decode gemini request");

    let thinking = ir::Request::from(&req).thinking.expect("thinking");
    assert_eq!(thinking, Effort::default());
}

#[test]
fn a_gemini_request_without_thinking_carries_none() {
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 100}
    });
    let req: gemini::Request = serde_json::from_value(body).expect("decode gemini request");

    assert!(ir::Request::from(&req).thinking.is_none());
}

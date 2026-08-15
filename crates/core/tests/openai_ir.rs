//! The IR ↔ OpenAI wire conversion, checked against the shapes OpenAI and its
//! compatible providers actually send and accept.
//!
//! These assert on serialized JSON rather than on Rust types on purpose: the
//! bug this suite exists for was a type that round-tripped through itself
//! perfectly and produced JSON no OpenAI endpoint would take.
use crabllm_core::{ChatCompletionRequest, ChatCompletionResponse, ir};
use serde_json::{Value, json};

fn ir_request(system: Option<Vec<ir::Content>>, messages: Vec<ir::Message>) -> ir::Request {
    ir::Request {
        model: "gpt-4o".into(),
        system,
        messages,
        max_tokens: 64,
        temperature: None,
        top_p: None,
        stop: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        stream: false,
    }
}

fn wire(req: &ir::Request) -> Value {
    serde_json::to_value(ChatCompletionRequest::from(req)).expect("serialize request")
}

fn reasoning(text: &str) -> ir::Content {
    ir::Content::Reasoning {
        text: text.into(),
        signature: Some("provider-private".into()),
    }
}

// ── Request direction: IR → OpenAI ──

#[test]
fn reasoning_never_reaches_the_wire() {
    let req = ir_request(
        Some(vec![
            reasoning("system scratchpad"),
            ir::Content::Text("be brief".into()),
        ]),
        vec![ir::Message {
            role: ir::Role::Assistant,
            content: vec![
                reasoning("assistant scratchpad"),
                ir::Content::Text("hi".into()),
            ],
        }],
    );

    let json = wire(&req).to_string();
    assert!(!json.contains("scratchpad"), "reasoning leaked: {json}");
    assert!(!json.contains("thinking"), "thinking block leaked: {json}");
    assert!(
        !json.contains("provider-private"),
        "signature leaked: {json}"
    );
}

#[test]
fn a_message_of_only_reasoning_is_dropped() {
    let req = ir_request(
        Some(vec![reasoning("system scratchpad")]),
        vec![ir::Message {
            role: ir::Role::Assistant,
            content: vec![reasoning("assistant scratchpad")],
        }],
    );

    assert_eq!(wire(&req)["messages"], json!([]));
}

#[test]
fn text_goes_as_a_bare_string() {
    let req = ir_request(
        None,
        vec![ir::Message {
            role: ir::Role::User,
            content: vec![ir::Content::Text("weather?".into())],
        }],
    );

    assert_eq!(
        wire(&req)["messages"],
        json!([{"role": "user", "content": "weather?"}])
    );
}

#[test]
fn a_tool_use_turn_matches_the_openai_shape() {
    let req = ir_request(
        None,
        vec![
            ir::Message {
                role: ir::Role::Assistant,
                content: vec![
                    reasoning("deciding to call the tool"),
                    ir::Content::ToolCall {
                        id: "call_1".into(),
                        name: "get_weather".into(),
                        input: json!({"city": "SF"}),
                    },
                ],
            },
            ir::Message {
                role: ir::Role::User,
                content: vec![ir::Content::ToolResult {
                    call_id: "call_1".into(),
                    content: vec![ir::Content::Text("72F".into())],
                }],
            },
        ],
    );

    // The assistant turn carries `tool_calls` with stringified arguments and
    // no content; the result is its own `role: "tool"` message keyed by id.
    assert_eq!(
        wire(&req)["messages"],
        json!([
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                }]
            },
            {"role": "tool", "content": "72F", "tool_call_id": "call_1"}
        ])
    );
}

#[test]
fn an_image_goes_as_a_data_url_part() {
    let req = ir_request(
        None,
        vec![ir::Message {
            role: ir::Role::User,
            content: vec![
                ir::Content::Text("what is this?".into()),
                ir::Content::Image {
                    media_type: "image/png".into(),
                    data: "AAAA".into(),
                },
            ],
        }],
    );

    assert_eq!(
        wire(&req)["messages"][0]["content"],
        json!([
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ])
    );
}

/// A tool message takes exactly `role`, `content` and `tool_call_id`. It has
/// no `name` — unlike system/user/assistant, which do. Checked against
/// `ChatCompletionToolMessageParam` in openai-python, where all three fields
/// are required and no `name` is defined, and against LiteLLM's
/// `ChatCompletionToolMessage`, which agrees.
#[test]
fn a_tool_message_carries_no_name() {
    let msg = serde_json::to_value(crabllm_core::Message::tool("call_1", "72F"))
        .expect("serialize tool message");

    assert_eq!(
        msg,
        json!({"role": "tool", "content": "72F", "tool_call_id": "call_1"})
    );
}

/// `content` is "required unless `tool_calls` or `function_call` is
/// specified" — so an assistant turn that only calls a tool omits it rather
/// than sending an empty string.
#[test]
fn an_assistant_tool_call_omits_content_entirely() {
    let req = ir_request(
        None,
        vec![ir::Message {
            role: ir::Role::Assistant,
            content: vec![ir::Content::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                input: json!({}),
            }],
        }],
    );

    let msg = &wire(&req)["messages"][0];
    assert!(
        msg.get("content").is_none(),
        "content should be absent, got {msg}"
    );
    assert!(msg.get("tool_calls").is_some());
}

// ── Request params ──

/// Streaming without `stream_options.include_usage` yields no usage chunk at
/// all, so the request would meter as zero tokens.
#[test]
fn streaming_asks_for_usage() {
    let mut req = ir_request(None, vec![]);
    req.stream = true;

    assert_eq!(wire(&req)["stream_options"], json!({"include_usage": true}));
}

#[test]
fn a_non_streaming_request_omits_stream_options() {
    let json = wire(&ir_request(None, vec![]));

    assert!(json.get("stream_options").is_none(), "got {json}");
}

/// Reasoning models reject `max_tokens`, `temperature` and `top_p`. Detection
/// follows LiteLLM: `o` + digit for the o-series, and `gpt-5` except the
/// `gpt-5-chat` line, which is an ordinary chat model.
#[test]
fn reasoning_models_get_max_completion_tokens() {
    for model in [
        "o1",
        "o3-mini",
        "o4-mini",
        "openai/o3",
        "gpt-5",
        "gpt-5-mini",
    ] {
        let mut req = ir_request(None, vec![]);
        req.model = model.into();
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        let json = wire(&req);

        assert_eq!(json["max_completion_tokens"], 64, "{model}");
        assert!(json.get("max_tokens").is_none(), "{model} kept max_tokens");
        assert!(
            json.get("temperature").is_none(),
            "{model} kept temperature"
        );
        assert!(json.get("top_p").is_none(), "{model} kept top_p");
    }
}

#[test]
fn ordinary_models_keep_max_tokens_and_sampling() {
    for model in [
        "gpt-4o",
        "gpt-5-chat-latest",
        "deepseek-chat",
        "o-not-a-digit",
    ] {
        let mut req = ir_request(None, vec![]);
        req.model = model.into();
        req.temperature = Some(0.7);
        let json = wire(&req);

        assert_eq!(json["max_tokens"], 64, "{model}");
        assert!(
            json.get("max_completion_tokens").is_none(),
            "{model} used max_completion_tokens"
        );
        assert_eq!(json["temperature"], 0.7, "{model}");
    }
}

/// A client may send `max_completion_tokens` itself, so reading a request back
/// into the IR has to accept either spelling.
#[test]
fn max_completion_tokens_is_read_back() {
    let body = json!({
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 512
    });
    let req: ChatCompletionRequest = serde_json::from_value(body).expect("decode request");

    assert_eq!(ir::Request::from(req).max_tokens, 512);
}

// ── Response direction: OpenAI → IR ──
//
// The payloads below are the response shapes OpenAI documents and DeepSeek
// mirrors. Before this suite existed, every one of them failed to deserialize
// with `invalid type: ..., expected a sequence`.

fn decode(body: Value) -> ir::Response {
    let resp: ChatCompletionResponse =
        serde_json::from_value(body).expect("decode chat completion response");
    ir::Response::from(resp)
}

fn envelope(message: Value, finish_reason: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
    })
}

#[test]
fn decodes_plain_string_content() {
    let ir_resp = decode(envelope(
        json!({"role": "assistant", "content": "Hello"}),
        "stop",
    ));

    assert!(matches!(
        ir_resp.content.as_slice(),
        [ir::Content::Text(t)] if t == "Hello"
    ));
    assert_eq!(ir_resp.usage.input_tokens, 9);
}

#[test]
fn decodes_null_content_with_tool_calls() {
    let ir_resp = decode(envelope(
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
            }]
        }),
        "tool_calls",
    ));

    match ir_resp.content.as_slice() {
        [ir::Content::ToolCall { id, name, input }] => {
            assert_eq!(id, "call_1");
            assert_eq!(name, "get_weather");
            assert_eq!(input, &json!({"city": "SF"}));
        }
        other => panic!("expected a single tool call, got {other:?}"),
    }
    assert!(matches!(ir_resp.stop_reason, Some(ir::StopReason::ToolUse)));
}

#[test]
fn decodes_reasoning_content() {
    let ir_resp = decode(envelope(
        json!({
            "role": "assistant",
            "content": "42",
            "reasoning_content": "thinking it through"
        }),
        "stop",
    ));

    assert!(ir_resp.content.iter().any(
        |c| matches!(c, ir::Content::Reasoning { text, .. } if text == "thinking it through")
    ));
    assert!(
        ir_resp
            .content
            .iter()
            .any(|c| matches!(c, ir::Content::Text(t) if t == "42"))
    );
}

#[test]
fn decodes_multi_part_content() {
    let ir_resp = decode(envelope(
        json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
        }),
        "stop",
    ));

    let texts: Vec<&str> = ir_resp
        .content
        .iter()
        .filter_map(|c| match c {
            ir::Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["first", "second"]);
}

/// An assistant content array may hold `refusal` parts alongside `text` —
/// `ContentArrayOfContentPart` in openai-python is exactly those two. An
/// unknown part type used to fail the whole decode.
#[test]
fn decodes_a_refusal_part() {
    let ir_resp = decode(envelope(
        json!({
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}]
        }),
        "stop",
    ));

    assert!(matches!(
        ir_resp.content.as_slice(),
        [ir::Content::Text(t)] if t == "I can't help with that."
    ));
}

/// `image_url` is an object carrying `url` and an optional `detail`, not a
/// bare string. `detail` must survive a decode/encode round trip.
#[test]
fn image_detail_survives_a_round_trip() {
    let body = json!({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.png", "detail": "high"}
            }]
        }]
    });

    let req: ChatCompletionRequest = serde_json::from_value(body).expect("decode request");
    let back = serde_json::to_value(&req).expect("re-encode request");

    assert_eq!(
        back["messages"][0]["content"][0]["image_url"],
        json!({"url": "https://example.com/a.png", "detail": "high"})
    );
}

/// OpenAI reports prompt cache hits under `prompt_tokens_details.cached_tokens`;
/// DeepSeek uses a flat `prompt_cache_hit_tokens`. Both have to be read, or a
/// cache hit meters as full-price input. This also covers the raw passthrough
/// path, whose billing peek in `usage.rs` decodes through the same type.
#[test]
fn openai_cached_tokens_are_billed_as_cache_reads() {
    let usage: crabllm_core::Usage = serde_json::from_value::<crabllm_core::OpenAiUsage>(json!({
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "prompt_tokens_details": {"cached_tokens": 900}
    }))
    .map(|u| (&u).into())
    .expect("decode usage");

    assert_eq!(usage.cache_read_tokens, 900);
    assert_eq!(
        usage.input_tokens, 100,
        "cached tokens must not double-count"
    );
}

#[test]
fn deepseek_cache_hits_still_work() {
    let usage: crabllm_core::Usage = serde_json::from_value::<crabllm_core::OpenAiUsage>(json!({
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "prompt_cache_hit_tokens": 900
    }))
    .map(|u| (&u).into())
    .expect("decode usage");

    assert_eq!(usage.cache_read_tokens, 900);
    assert_eq!(usage.input_tokens, 100);
}

/// The same peek the proxy runs over raw SSE bytes for billing.
#[test]
fn the_billing_peek_sees_cached_tokens() {
    let chunk = json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
        "model": "gpt-4o", "choices": [],
        "usage": {
            "prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 900}
        }
    })
    .to_string();

    let usage = crabllm_core::Usage::from(chunk.as_bytes());

    assert_eq!(usage.cache_read_tokens, 900);
    assert_eq!(usage.input_tokens, 100);
}

#[test]
fn unmodelled_fields_survive_a_round_trip() {
    let body = envelope(
        json!({"role": "assistant", "content": "hi", "refusal": null, "annotations": []}),
        "stop",
    );
    let resp: ChatCompletionResponse =
        serde_json::from_value(body).expect("decode chat completion response");
    let message = serde_json::to_value(&resp.choices[0].message).expect("re-serialize");

    assert_eq!(message["annotations"], json!([]));
}

// ── A full turn survives the round trip ──

#[test]
fn a_tool_turn_round_trips_through_the_ir() {
    let original = ir_request(
        None,
        vec![
            ir::Message {
                role: ir::Role::User,
                content: vec![ir::Content::Text("weather?".into())],
            },
            ir::Message {
                role: ir::Role::Assistant,
                content: vec![ir::Content::ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    input: json!({"city": "SF"}),
                }],
            },
            ir::Message {
                role: ir::Role::User,
                content: vec![ir::Content::ToolResult {
                    call_id: "call_1".into(),
                    content: vec![ir::Content::Text("72F".into())],
                }],
            },
        ],
    );

    let back = ir::Request::from(ChatCompletionRequest::from(&original));

    assert_eq!(back.messages.len(), 3);
    assert!(matches!(
        back.messages[1].content.as_slice(),
        [ir::Content::ToolCall { id, .. }] if id == "call_1"
    ));
    assert!(matches!(
        back.messages[2].content.as_slice(),
        [ir::Content::ToolResult { call_id, .. }] if call_id == "call_1"
    ));
}

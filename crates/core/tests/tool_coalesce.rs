//! Regression: OpenAI-shaped parallel tool calls arrive as one user message
//! per `tool_result`. Anthropic requires them merged into the single user
//! message after the assistant, so the IR→Anthropic translation must coalesce
//! them. Before this was wired in, the split reached Anthropic and was rejected.
use crabllm_core::{AnthropicRequest, ir};

fn tool_result(id: &str) -> ir::Message {
    ir::Message {
        role: ir::Role::User,
        content: vec![ir::Content::ToolResult {
            call_id: id.into(),
            content: vec![ir::Content::Text("ok".into())],
        }],
    }
}

#[test]
fn parallel_tool_results_coalesce_into_one_user_message() {
    let req = ir::Request {
        model: "claude".into(),
        system: None,
        // assistant turn, then two separate tool_result user messages.
        messages: vec![
            ir::Message::assistant("using tools"),
            tool_result("t1"),
            tool_result("t2"),
        ],
        max_tokens: 64,
        temperature: None,
        top_p: None,
        stop: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        stream: false,
    };

    let anthropic = AnthropicRequest::from(&req);

    // assistant + a single merged user message — not assistant + two users.
    assert_eq!(
        anthropic.messages.len(),
        2,
        "split parallel tool_results should coalesce into one user message"
    );
    assert_eq!(
        anthropic
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .count(),
        1
    );
}

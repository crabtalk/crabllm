//! Regression tests for IR to OpenAI request conversion.
use crabllm_core::{ChatCompletionRequest, ContentBlock, ToolResultContent, ir};

fn reasoning(text: &str) -> ir::Content {
    ir::Content::Reasoning {
        text: text.into(),
        signature: Some("provider-private".into()),
    }
}

fn request(system: Option<Vec<ir::Content>>, messages: Vec<ir::Message>) -> ir::Request {
    ir::Request {
        model: "openai-compatible".into(),
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

fn contains_thinking(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Thinking { .. } => true,
        ContentBlock::ToolResult {
            content: ToolResultContent::Blocks(blocks),
            ..
        } => blocks.iter().any(contains_thinking),
        _ => false,
    }
}

#[test]
fn strips_reasoning_from_system_messages_and_nested_tool_results() {
    let req = request(
        Some(vec![
            reasoning("system scratchpad"),
            ir::Content::Text("system".into()),
        ]),
        vec![
            ir::Message {
                role: ir::Role::Assistant,
                content: vec![
                    reasoning("assistant scratchpad"),
                    ir::Content::Text("answer".into()),
                ],
            },
            ir::Message {
                role: ir::Role::User,
                content: vec![ir::Content::ToolResult {
                    call_id: "tool-1".into(),
                    content: vec![
                        reasoning("nested scratchpad"),
                        ir::Content::Text("result".into()),
                    ],
                }],
            },
        ],
    );

    let openai = ChatCompletionRequest::from(&req);

    assert_eq!(openai.messages.len(), 3);
    assert!(
        openai
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .all(|block| !contains_thinking(block))
    );
}

#[test]
fn drops_messages_that_contain_only_reasoning() {
    let req = request(
        Some(vec![reasoning("system scratchpad")]),
        vec![ir::Message {
            role: ir::Role::Assistant,
            content: vec![reasoning("assistant scratchpad")],
        }],
    );

    let openai = ChatCompletionRequest::from(&req);

    assert!(openai.messages.is_empty());
}

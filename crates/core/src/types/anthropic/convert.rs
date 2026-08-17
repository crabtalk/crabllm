//! Conversions into the Anthropic tool wire shape.

use crate::types::{
    anthropic::{Content, ContentBlock, Message, Tool},
    ir, openai,
};
use serde_json::{Value, json};

impl From<&openai::Tool> for Tool {
    fn from(tool: &openai::Tool) -> Self {
        Self {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            // Anthropic rejects a tool without an object schema, so a
            // parameterless tool still declares an empty object.
            input_schema: tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({"type": "object"})),
            cache_control: None,
        }
    }
}

impl Tool {
    /// Convert a tool set for the Messages API, marking the last one as a
    /// prompt-cache breakpoint.
    ///
    /// Anthropic caches everything *up to and including* a marked block,
    /// so one marker on the last tool covers the whole declaration — which
    /// is stable across a conversation and worth reusing. Marking each
    /// tool instead would spend breakpoints (there are four) for nothing.
    pub fn cached(tools: &[openai::Tool]) -> Vec<Self> {
        let last = tools.len().saturating_sub(1);
        tools
            .iter()
            .enumerate()
            .map(|(i, tool)| Self {
                cache_control: (i == last).then(|| json!({"type": "ephemeral"})),
                ..Self::from(tool)
            })
            .collect()
    }
}

/// Encode a tool choice for the Messages API.
///
/// Deliberately not `From<&openai::ToolChoice> for Value`: that conversion
/// is already taken by the OpenAI encoding (`"auto"`, `"none"`, …), and a
/// second one to the same target would resolve by reference-ness, which is
/// no way to pick a wire format.
pub fn tool_choice(choice: &openai::ToolChoice) -> Value {
    match choice {
        openai::ToolChoice::Auto => json!({"type": "auto"}),
        openai::ToolChoice::Required => json!({"type": "any"}),
        openai::ToolChoice::Function { name } => json!({"type": "tool", "name": name}),
        openai::ToolChoice::Disabled => json!({"type": "none"}),
    }
}

impl Message {
    /// Assemble an assistant message from decoded parts.
    ///
    /// A text block is emitted even when empty unless there are tool calls to
    /// carry the turn — a message with no content at all is not a valid turn.
    pub fn assistant_parts(
        text: impl Into<String>,
        reasoning: Option<String>,
        tool_calls: &[openai::ToolCall],
    ) -> Self {
        let mut blocks = Vec::new();
        if let Some(thinking) = reasoning.filter(|s| !s.is_empty()) {
            blocks.push(ContentBlock::Thinking {
                thinking,
                signature: None,
            });
        }
        let text: String = text.into();
        if !text.is_empty() || tool_calls.is_empty() {
            blocks.push(ContentBlock::text(text));
        }
        for call in tool_calls {
            blocks.push(ContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.function.name.clone(),
                input: crate::json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| json!({})),
                cache_control: None,
            });
        }
        Self {
            role: "assistant".to_string(),
            content: Content::Blocks(blocks),
        }
    }

    /// This message's `tool_use` blocks in the OpenAI tool-call shape — the
    /// dialect-neutral currency for dispatching a tool.
    pub fn tool_calls(&self) -> Vec<openai::ToolCall> {
        self.blocks()
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => Some(openai::ToolCall {
                    index: None,
                    id: id.clone(),
                    kind: openai::ToolType::Function,
                    function: openai::FunctionCall {
                        name: name.clone(),
                        arguments: crate::json::to_string(input).unwrap_or_default(),
                    },
                }),
                _ => None,
            })
            .collect()
    }
}

/// Decode a `stop_reason` from the Messages API into the OpenAI finish reason.
///
/// Not `From<&str>`: that conversion is already taken by the OpenAI encoding
/// (`"stop"`, `"length"`, …), and a second one over the same source type
/// would make the dialect invisible at the call site.
pub fn finish_reason(reason: &str) -> openai::FinishReason {
    match reason {
        "end_turn" => openai::FinishReason::Stop,
        "max_tokens" => openai::FinishReason::Length,
        "tool_use" => openai::FinishReason::ToolCalls,
        other => openai::FinishReason::Custom(other.to_string()),
    }
}

/// Decode a `stop_reason` from the Messages API into the canonical IR.
///
/// The IR has no open variant, so anything unrecognised lands on `End` — the
/// turn is over either way.
pub fn stop_reason(reason: &str) -> ir::StopReason {
    match reason {
        "max_tokens" => ir::StopReason::MaxTokens,
        "tool_use" => ir::StopReason::ToolUse,
        _ => ir::StopReason::End,
    }
}

use crate::{
    FunctionCall, ToolCall, ToolType,
    types::anthropic::{self, Message},
};
use std::collections::BTreeMap;

/// Accumulates a streamed Anthropic response back into a whole [`Message`].
///
/// A caller that streams for the deltas still needs the finished message —
/// to append to a conversation, to persist, to send back on the next turn.
/// Feed every event to [`accept`](Self::accept) and [`build`](Self::build)
/// the result at `message_stop`.
#[derive(Debug, Default)]
pub struct MessageBuilder {
    content: String,
    reasoning: String,
    /// Keyed by block index — tool arguments arrive as JSON fragments split
    /// across deltas, and interleaved across blocks.
    tool_blocks: BTreeMap<u32, (String, String, String)>,
}

impl MessageBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn accept(&mut self, event: &anthropic::StreamEvent) {
        match event {
            anthropic::StreamEvent::ContentBlockStart {
                index,
                content_block: anthropic::ContentBlock::ToolUse { id, name, .. },
            } => {
                self.tool_blocks
                    .insert(*index, (id.clone(), name.clone(), String::new()));
            }
            anthropic::StreamEvent::ContentBlockDelta { index, delta } => match delta {
                anthropic::BlockDelta::Text { text } => self.content.push_str(text),
                anthropic::BlockDelta::Thinking { thinking } => self.reasoning.push_str(thinking),
                anthropic::BlockDelta::InputJson { partial_json } => {
                    if let Some((_, _, args)) = self.tool_blocks.get_mut(index) {
                        args.push_str(partial_json);
                    }
                }
            },
            _ => {}
        }
    }

    /// The tool calls so far, with arguments exactly as they arrived — a
    /// half-streamed one reads back as a partial JSON string here, where
    /// [`build`](Self::build) would parse it to an empty object.
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.tool_blocks
            .iter()
            .filter(|(_, (id, name, _))| !id.is_empty() && !name.is_empty())
            .map(|(idx, (id, name, args))| ToolCall {
                index: Some(*idx),
                id: id.clone(),
                kind: ToolType::Function,
                function: FunctionCall {
                    name: name.clone(),
                    arguments: args.clone(),
                },
            })
            .collect()
    }

    /// Names of the tool calls whose accumulated arguments aren't valid JSON.
    /// Non-empty after a complete stream means the model emitted a tool call
    /// we cannot dispatch.
    pub fn malformed_tool_calls(&self) -> Vec<&str> {
        self.tool_blocks
            .values()
            .filter(|(id, name, _)| !id.is_empty() && !name.is_empty())
            .filter(|(_, _, args)| crate::json::from_str::<serde_json::Value>(args).is_err())
            .map(|(_, name, _)| name.as_str())
            .collect()
    }

    pub fn build(self) -> Message {
        let tool_calls = self.tool_calls();
        Message::assistant_parts(self.content, Some(self.reasoning), &tool_calls)
    }
}

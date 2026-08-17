use crate::{
    Usage,
    ir::{self, Content, Message, Role, StopReason, StreamEvent},
    types::anthropic::{
        self, BlockDelta, ContentBlock, Messages, ThinkingConfig, ToolResultContent,
    },
};

impl From<crate::anthropic::Request> for ir::Request {
    fn from(req: crate::anthropic::Request) -> Self {
        let system = req.system.map(|s| match s {
            anthropic::System::Text(text) => vec![Content::Text(text)],
            anthropic::System::Blocks(blocks) => blocks_to_ir(blocks),
        });

        let messages = req
            .messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role.as_str() {
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                let blocks = match msg.content {
                    anthropic::Content::Text(s) => vec![Content::Text(s)],
                    anthropic::Content::Blocks(b) => blocks_to_ir(b),
                };
                Message {
                    role,
                    content: blocks,
                }
            })
            .collect();

        let tools = req.tools.map(|tools| {
            tools
                .into_iter()
                .map(|t| ir::Tool {
                    name: t.name,
                    description: t.description,
                    parameters: Some(t.input_schema),
                    cache_control: t.cache_control,
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().and_then(|v| {
            let kind = v.get("type")?.as_str()?;
            Some(match kind {
                "auto" => ir::ToolChoice::Auto,
                "any" => ir::ToolChoice::Required,
                "none" => ir::ToolChoice::Disabled,
                "tool" => ir::ToolChoice::Named(v.get("name")?.as_str()?.to_string()),
                _ => return None,
            })
        });

        let thinking = req.thinking.map(|t| ir::Thinking {
            effort: if t.kind == "disabled" {
                ir::Effort::None
            } else {
                // An enabled config without a budget is Anthropic's floor.
                t.budget_tokens
                    .map(ir::Effort::from_budget)
                    .unwrap_or(ir::Effort::Low)
            },
            budget_tokens: t.budget_tokens,
        });

        ir::Request {
            model: req.model,
            system,
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop_sequences,
            tools,
            tool_choice,
            thinking,
            stream: req.stream.unwrap_or(false),
        }
    }
}

impl From<&ir::Request> for crate::anthropic::Request {
    fn from(req: &ir::Request) -> Self {
        let system = req
            .system
            .as_ref()
            .map(|blocks| anthropic::System::Blocks(ir_to_blocks(blocks)));

        let mut messages: Vec<anthropic::Message> = req
            .messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::Assistant => "assistant".to_string(),
                    Role::User | Role::System => "user".to_string(),
                };
                anthropic::Message {
                    role,
                    content: anthropic::Content::Blocks(ir_to_blocks(&msg.content)),
                }
            })
            .collect();
        // Parallel tool calls arrive as one user message per tool_result; Anthropic
        // requires them merged into the single user message after the assistant.
        messages.coalesce_tool_results();
        messages.ensure_tool_pairing();

        let tools = req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| anthropic::Tool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t
                        .parameters
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({})),
                    cache_control: t.cache_control.clone(),
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
            ir::ToolChoice::Auto => serde_json::json!({"type": "auto"}),
            ir::ToolChoice::Required => serde_json::json!({"type": "any"}),
            ir::ToolChoice::Disabled => serde_json::json!({"type": "none"}),
            ir::ToolChoice::Named(name) => serde_json::json!({"type": "tool", "name": name}),
        });

        let thinking = req.thinking.as_ref().map(|t| {
            let off = t.effort == ir::Effort::None;
            ThinkingConfig {
                kind: if off { "disabled" } else { "enabled" }.to_string(),
                budget_tokens: (!off).then(|| t.budget(req.max_tokens)),
            }
        });

        crate::anthropic::Request {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            system,
            temperature: req.temperature,
            top_p: req.top_p,
            stream: if req.stream { Some(true) } else { None },
            tools,
            tool_choice,
            stop_sequences: req.stop.clone(),
            thinking,
        }
    }
}

impl From<anthropic::Response> for ir::Response {
    fn from(resp: anthropic::Response) -> Self {
        let stop_reason = resp.stop_reason.as_deref().map(anthropic::stop_reason);

        ir::Response {
            id: resp.id,
            model: resp.model,
            content: resp.content.into_iter().map(Content::from).collect(),
            stop_reason,
            usage: crate::Usage::from(&resp.usage),
        }
    }
}

impl From<&ir::Response> for anthropic::Response {
    fn from(resp: &ir::Response) -> Self {
        let stop_reason = resp.stop_reason.as_ref().map(|r| match r {
            StopReason::End => "end_turn".to_string(),
            StopReason::MaxTokens => "max_tokens".to_string(),
            StopReason::ToolUse => "tool_use".to_string(),
        });

        anthropic::Response {
            id: resp.id.clone(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            model: resp.model.clone(),
            content: resp.content.iter().map(ContentBlock::from).collect(),
            stop_reason,
            stop_sequence: None,
            usage: anthropic::Usage::from(&resp.usage),
        }
    }
}

impl anthropic::StreamEvent {
    pub fn to_ir_events(&self) -> Vec<StreamEvent> {
        match self {
            anthropic::StreamEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse { id, name, .. },
                ..
            } => vec![StreamEvent::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
            }],
            anthropic::StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                BlockDelta::Text { text } if !text.is_empty() => {
                    vec![StreamEvent::TextDelta(text.clone())]
                }
                BlockDelta::Thinking { thinking } if !thinking.is_empty() => {
                    vec![StreamEvent::ReasoningDelta(thinking.clone())]
                }
                BlockDelta::InputJson { partial_json } if !partial_json.is_empty() => {
                    vec![StreamEvent::ToolCallDelta(partial_json.clone())]
                }
                _ => vec![],
            },
            anthropic::StreamEvent::MessageDelta { delta, usage } => {
                let mut events = vec![StreamEvent::Usage(Usage::from(usage))];
                if let Some(reason) = &delta.stop_reason {
                    events.push(StreamEvent::Stop(anthropic::stop_reason(reason)));
                }
                events
            }
            _ => vec![],
        }
    }
}

impl From<ContentBlock> for Content {
    fn from(block: ContentBlock) -> Self {
        match block {
            ContentBlock::Text { text, .. } => Content::Text(text),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Content::ToolCall { id, name, input },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => Content::ToolResult {
                call_id: tool_use_id,
                content: match content {
                    ToolResultContent::Text(s) => vec![Content::Text(s)],
                    ToolResultContent::Blocks(b) => b.into_iter().map(Content::from).collect(),
                },
            },
            ContentBlock::Thinking {
                thinking,
                signature,
            } => Content::Reasoning {
                text: thinking,
                signature,
            },
            ContentBlock::Image { source, .. } => {
                let media_type = source
                    .get("media_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("application/octet-stream")
                    .to_string();
                let data = source
                    .get("data")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                Content::Image { media_type, data }
            }
        }
    }
}

impl From<&Content> for ContentBlock {
    fn from(content: &Content) -> Self {
        match content {
            Content::Text(text) => ContentBlock::Text {
                text: text.clone(),
                cache_control: None,
            },
            Content::ToolCall { id, name, input } => ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                cache_control: None,
            },
            Content::ToolResult { call_id, content } => ContentBlock::ToolResult {
                tool_use_id: call_id.clone(),
                name: None,
                content: if content.len() == 1 {
                    if let Some(Content::Text(s)) = content.first() {
                        ToolResultContent::Text(s.clone())
                    } else {
                        ToolResultContent::Blocks(content.iter().map(ContentBlock::from).collect())
                    }
                } else {
                    ToolResultContent::Blocks(content.iter().map(ContentBlock::from).collect())
                },
                cache_control: None,
            },
            Content::Reasoning { text, signature } => ContentBlock::Thinking {
                thinking: text.clone(),
                signature: signature.clone(),
            },
            // Handled by `ir_to_blocks`, which folds it onto the previous
            // block. Reached only if a breakpoint is converted in isolation.
            Content::CacheBreakpoint => ContentBlock::text(""),
            Content::Image { media_type, data } => ContentBlock::Image {
                source: serde_json::json!({
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                }),
                cache_control: None,
            },
        }
    }
}

/// Anthropic blocks to IR content: a block carrying `cache_control` is
/// followed by a [`Content::CacheBreakpoint`], since the mark describes the
/// prefix ending at that block rather than the block itself.
fn blocks_to_ir(blocks: Vec<ContentBlock>) -> Vec<Content> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        let breakpoint = block.is_cache_breakpoint();
        out.push(Content::from(block));
        if breakpoint {
            out.push(Content::CacheBreakpoint);
        }
    }
    out
}

/// The inverse: a breakpoint folds onto the block it follows. A breakpoint
/// with nothing before it has no block to mark and is dropped.
fn ir_to_blocks(contents: &[Content]) -> Vec<ContentBlock> {
    let mut out: Vec<ContentBlock> = Vec::with_capacity(contents.len());
    for content in contents {
        if matches!(content, Content::CacheBreakpoint) {
            if let Some(last) = out.last_mut() {
                last.mark_cache_breakpoint();
            }
            continue;
        }
        out.push(ContentBlock::from(content));
    }
    out
}

use crate::{
    AnthropicContent, AnthropicMessage, AnthropicResponse, AnthropicStreamEvent, AnthropicSystem,
    AnthropicTool, AnthropicUsage, BlockDelta, ContentBlock, ThinkingConfig, Usage,
    ir::{self, Content, Message, Role, StopReason, StreamEvent},
};

impl From<crate::AnthropicRequest> for ir::Request {
    fn from(req: crate::AnthropicRequest) -> Self {
        let system = req.system.map(|s| match s {
            AnthropicSystem::Text(text) => vec![Content::Text(text)],
            AnthropicSystem::Blocks(blocks) => blocks.into_iter().map(Content::from).collect(),
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
                    AnthropicContent::Text(s) => vec![Content::Text(s)],
                    AnthropicContent::Blocks(b) => b.into_iter().map(Content::from).collect(),
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

impl From<&ir::Request> for crate::AnthropicRequest {
    fn from(req: &ir::Request) -> Self {
        let system = req
            .system
            .as_ref()
            .map(|blocks| AnthropicSystem::Blocks(blocks.iter().map(ContentBlock::from).collect()));

        let messages = req
            .messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::Assistant => "assistant".to_string(),
                    Role::User | Role::System => "user".to_string(),
                };
                AnthropicMessage {
                    role,
                    content: AnthropicContent::Blocks(
                        msg.content.iter().map(ContentBlock::from).collect(),
                    ),
                }
            })
            .collect();

        let tools = req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| AnthropicTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t
                        .parameters
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({})),
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
            ir::ToolChoice::Auto => serde_json::json!({"type": "auto"}),
            ir::ToolChoice::Required => serde_json::json!({"type": "any"}),
            ir::ToolChoice::Disabled => serde_json::json!({"type": "none"}),
            ir::ToolChoice::Named(name) => serde_json::json!({"type": "tool", "name": name}),
        });

        let thinking = req.thinking.as_ref().map(|t| ThinkingConfig {
            kind: "enabled".to_string(),
            budget_tokens: t.budget_tokens,
        });

        crate::AnthropicRequest {
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

impl From<AnthropicResponse> for ir::Response {
    fn from(resp: AnthropicResponse) -> Self {
        let stop_reason = resp.stop_reason.as_deref().map(|r| match r {
            "end_turn" => StopReason::End,
            "max_tokens" => StopReason::MaxTokens,
            "tool_use" => StopReason::ToolUse,
            _ => StopReason::End,
        });

        ir::Response {
            id: resp.id,
            model: resp.model,
            content: resp.content.into_iter().map(Content::from).collect(),
            stop_reason,
            usage: crate::Usage::from(&resp.usage),
        }
    }
}

impl From<&ir::Response> for AnthropicResponse {
    fn from(resp: &ir::Response) -> Self {
        let stop_reason = resp.stop_reason.as_ref().map(|r| match r {
            StopReason::End => "end_turn".to_string(),
            StopReason::MaxTokens => "max_tokens".to_string(),
            StopReason::ToolUse => "tool_use".to_string(),
        });

        AnthropicResponse {
            id: resp.id.clone(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            model: resp.model.clone(),
            content: resp.content.iter().map(ContentBlock::from).collect(),
            stop_reason,
            stop_sequence: None,
            usage: AnthropicUsage::from(&resp.usage),
        }
    }
}

impl AnthropicStreamEvent {
    pub fn to_ir_events(&self) -> Vec<StreamEvent> {
        match self {
            AnthropicStreamEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse { id, name, .. },
                ..
            } => vec![StreamEvent::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
            }],
            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => match delta {
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
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let mut events = vec![StreamEvent::Usage(Usage::from(usage))];
                if let Some(reason) = &delta.stop_reason {
                    let stop = match reason.as_str() {
                        "end_turn" => StopReason::End,
                        "max_tokens" => StopReason::MaxTokens,
                        "tool_use" => StopReason::ToolUse,
                        _ => StopReason::End,
                    };
                    events.push(StreamEvent::Stop(stop));
                }
                events
            }
            _ => vec![],
        }
    }
}

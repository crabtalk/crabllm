use crate::{
    ChatCompletionResponse, ContentBlock, FinishReason, OpenAiUsage, ToolResultContent, Usage,
    ir::{self, Content, Message, Role, StopReason},
};

impl From<crate::ChatCompletionRequest> for ir::Request {
    fn from(req: crate::ChatCompletionRequest) -> Self {
        let mut system = None;
        let mut messages = Vec::with_capacity(req.messages.len());

        for msg in req.messages {
            match msg.role {
                crate::Role::System | crate::Role::Developer => {
                    let blocks: Vec<Content> =
                        msg.content.into_iter().map(Content::from).collect();
                    match &mut system {
                        Some(existing) => {
                            let v: &mut Vec<Content> = existing;
                            v.extend(blocks);
                        }
                        None => system = Some(blocks),
                    }
                }
                crate::Role::Assistant => {
                    messages.push(Message {
                        role: Role::Assistant,
                        content: msg.content.into_iter().map(Content::from).collect(),
                    });
                }
                _ => {
                    messages.push(Message {
                        role: Role::User,
                        content: msg.content.into_iter().map(Content::from).collect(),
                    });
                }
            }
        }

        let stop = req.stop.map(|s| match s {
            crate::Stop::Single(s) => vec![s],
            crate::Stop::Multiple(v) => v,
        });

        let tools = req.tools.map(|tools| {
            tools
                .into_iter()
                .map(|t| ir::Tool {
                    name: t.function.name,
                    description: t.function.description,
                    parameters: t.function.parameters,
                })
                .collect()
        });

        let tool_choice = req.tool_choice.map(|tc| match tc {
            crate::ToolChoice::Auto => ir::ToolChoice::Auto,
            crate::ToolChoice::Required => ir::ToolChoice::Required,
            crate::ToolChoice::Disabled => ir::ToolChoice::Disabled,
            crate::ToolChoice::Function { name } => ir::ToolChoice::Named(name),
        });

        let thinking = req.thinking.map(|t| ir::Thinking {
            budget_tokens: t.budget_tokens,
        });

        ir::Request {
            model: req.model,
            system,
            messages,
            max_tokens: req.max_tokens.or(req.anthropic_max_tokens).unwrap_or(4096),
            temperature: req.temperature,
            top_p: req.top_p,
            stop,
            tools,
            tool_choice,
            thinking,
            stream: req.stream.unwrap_or(false),
        }
    }
}

impl From<ChatCompletionResponse> for ir::Response {
    fn from(resp: ChatCompletionResponse) -> Self {
        let (content, stop_reason) = resp
            .choices
            .into_iter()
            .next()
            .map(|choice| {
                let stop = choice.finish_reason.as_ref().map(|r| match r {
                    FinishReason::Stop => StopReason::End,
                    FinishReason::Length => StopReason::MaxTokens,
                    FinishReason::ToolCalls => StopReason::ToolUse,
                    _ => StopReason::End,
                });
                let content = choice
                    .message
                    .content
                    .into_iter()
                    .map(Content::from)
                    .collect();
                (content, stop)
            })
            .unwrap_or_default();

        let usage = resp.usage.as_ref().map(Usage::from).unwrap_or_default();

        ir::Response {
            id: resp.id,
            model: resp.model,
            content,
            stop_reason,
            usage,
        }
    }
}

impl From<ContentBlock> for Content {
    fn from(block: ContentBlock) -> Self {
        match block {
            ContentBlock::Text { text, .. } => Content::Text(text),
            ContentBlock::ToolUse { id, name, input, .. } => Content::ToolCall { id, name, input },
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

impl From<&ir::Response> for ChatCompletionResponse {
    fn from(resp: &ir::Response) -> Self {
        let finish_reason = resp.stop_reason.as_ref().map(|r| match r {
            StopReason::End => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::ToolUse => FinishReason::ToolCalls,
        });

        Self {
            id: resp.id.clone(),
            object: "chat.completion".to_string(),
            created: 0,
            model: resp.model.clone(),
            choices: vec![crate::Choice {
                index: 0,
                message: crate::Message {
                    role: crate::Role::Assistant,
                    content: resp.content.iter().map(ContentBlock::from).collect(),
                },
                finish_reason,
                logprobs: None,
            }],
            usage: Some(OpenAiUsage::from(&resp.usage)),
            system_fingerprint: None,
        }
    }
}

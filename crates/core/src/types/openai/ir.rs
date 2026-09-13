use crate::{
    ChatCompletionChunk, ChatCompletionResponse, ContentPart, FinishReason, FunctionCall,
    FunctionDef, ImageUrl, MessageContent, OpenAiUsage, ToolType, Usage,
    ir::{self, Content, Message, Role, StopReason, StreamEvent},
};
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

impl From<crate::ChatCompletionRequest> for ir::Request {
    fn from(req: crate::ChatCompletionRequest) -> Self {
        let mut system: Option<Vec<Content>> = None;
        let mut messages = Vec::with_capacity(req.messages.len());

        for msg in req.messages {
            match msg.role {
                crate::Role::System | crate::Role::Developer => {
                    let blocks = message_to_ir(&msg);
                    system.get_or_insert_default().extend(blocks);
                }
                // A `role: "tool"` message is a tool result keyed by
                // `tool_call_id`; the IR carries it inside the following user
                // turn instead of as a role of its own.
                crate::Role::Tool => {
                    messages.push(Message {
                        role: Role::User,
                        content: vec![Content::ToolResult {
                            call_id: msg.tool_call_id.clone().unwrap_or_default(),
                            content: message_to_ir(&msg),
                        }],
                    });
                }
                crate::Role::Assistant => {
                    messages.push(Message {
                        role: Role::Assistant,
                        content: message_to_ir(&msg),
                    });
                }
                _ => {
                    messages.push(Message {
                        role: Role::User,
                        content: message_to_ir(&msg),
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
                    cache_control: t.cache_control,
                })
                .collect()
        });

        let tool_choice = req.tool_choice.map(|tc| match tc {
            crate::ToolChoice::Auto => ir::ToolChoice::Auto,
            crate::ToolChoice::Required => ir::ToolChoice::Required,
            crate::ToolChoice::Disabled => ir::ToolChoice::Disabled,
            crate::ToolChoice::Function { name } => ir::ToolChoice::Named(name),
        });

        let thinking = req.reasoning_effort.as_deref().and_then(|s| s.parse().ok());

        ir::Request {
            model: req.model,
            system,
            messages,
            max_tokens: req
                .max_tokens
                .or(req.max_completion_tokens)
                .or(req.anthropic_max_tokens)
                .unwrap_or(4096),
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
                (message_to_ir(&choice.message), stop)
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

impl From<&ir::Request> for crate::ChatCompletionRequest {
    fn from(req: &ir::Request) -> Self {
        let mut messages = Vec::new();

        if let Some(system) = &req.system
            && let Some(content) = openai_content(system)
        {
            messages.push(crate::Message {
                role: crate::Role::System,
                content: Some(content),
                ..Default::default()
            });
        }

        for msg in &req.messages {
            let role = match msg.role {
                Role::System => crate::Role::System,
                Role::User => crate::Role::User,
                Role::Assistant => crate::Role::Assistant,
            };

            // Reasoning is the model's private scratchpad and has no slot on
            // this wire, so it is dropped. Tool calls lift onto `tool_calls`,
            // and tool results leave as their own `role: "tool"` messages.
            let tool_calls: Vec<crate::ToolCall> = msg
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::ToolCall { id, name, input } => Some(crate::ToolCall {
                        index: None,
                        id: id.clone(),
                        kind: ToolType::Function,
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: input.to_string(),
                        },
                    }),
                    _ => None,
                })
                .collect();
            let content = openai_content(&msg.content);

            if content.is_some() || !tool_calls.is_empty() {
                messages.push(crate::Message {
                    role,
                    content,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    ..Default::default()
                });
            }

            for c in &msg.content {
                if let Content::ToolResult { call_id, content } = c {
                    messages.push(crate::Message {
                        role: crate::Role::Tool,
                        content: Some(MessageContent::Text(ir_text(content))),
                        tool_call_id: Some(call_id.clone()),
                        ..Default::default()
                    });
                }
            }
        }

        let stop = req.stop.as_ref().map(|seqs| {
            if seqs.len() == 1 {
                crate::Stop::Single(seqs[0].clone())
            } else {
                crate::Stop::Multiple(seqs.clone())
            }
        });

        let tools = req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| crate::Tool {
                    kind: ToolType::Function,
                    function: FunctionDef {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                    strict: None,
                    cache_control: None,
                })
                .collect()
        });

        let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
            ir::ToolChoice::Auto => crate::ToolChoice::Auto,
            ir::ToolChoice::Required => crate::ToolChoice::Required,
            ir::ToolChoice::Disabled => crate::ToolChoice::Disabled,
            ir::ToolChoice::Named(name) => crate::ToolChoice::Function { name: name.clone() },
        });

        // Reasoning models take `max_completion_tokens` and reject sampling
        // params outright, so sending the usual shape is a 400 rather than a
        // degraded answer.
        let reasoning = is_reasoning_model(&req.model);

        crate::ChatCompletionRequest {
            model: req.model.clone(),
            messages,
            temperature: (!reasoning).then_some(req.temperature).flatten(),
            top_p: (!reasoning).then_some(req.top_p).flatten(),
            max_tokens: (!reasoning).then_some(req.max_tokens),
            max_completion_tokens: reasoning.then_some(req.max_tokens),
            stream: if req.stream { Some(true) } else { None },
            // Streaming without this yields no usage chunk, so the request
            // would meter as zero tokens.
            stream_options: req.stream.then_some(crate::StreamOptions {
                include_usage: true,
            }),
            stop,
            tools,
            tool_choice,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            user: None,
            reasoning_effort: req.thinking.map(|effort| effort.to_string()),
            anthropic_max_tokens: Some(req.max_tokens),
            extra: serde_json::Map::new(),
        }
    }
}

/// The text/image half of IR content as OpenAI message content. Tool calls,
/// tool results and reasoning have no place here — the caller lifts those onto
/// `tool_calls`, separate `role: "tool"` messages, and nothing respectively.
/// `None` when there is nothing to send.
fn openai_content(contents: &[Content]) -> Option<MessageContent> {
    let parts: Vec<ContentPart> = contents
        .iter()
        .filter_map(|c| match c {
            Content::Text(text) if !text.is_empty() => {
                Some(ContentPart::Text { text: text.clone() })
            }
            Content::Image { media_type, data } => Some(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:{media_type};base64,{data}"),
                    detail: None,
                },
            }),
            _ => None,
        })
        .collect();

    match parts.as_slice() {
        [] => None,
        // A lone text part goes as a bare string — the shape every
        // OpenAI-compatible endpoint accepts, including the older ones.
        [ContentPart::Text { text }] => Some(MessageContent::Text(text.clone())),
        _ => Some(MessageContent::Parts(parts)),
    }
}

/// Whether a model is a reasoning model, which rejects `max_tokens` in favour
/// of `max_completion_tokens` and rejects `temperature`/`top_p` outright.
///
/// Two families, matching how LiteLLM detects them: the o-series (`o` followed
/// by a digit — o1, o3, o4-mini) and gpt-5, minus the `gpt-5-chat` line, which
/// is an ordinary chat model. A provider prefix is stripped first, so both
/// `o3-mini` and `openai/o3-mini` match.
fn is_reasoning_model(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    let mut chars = name.chars();
    let o_series = chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit());
    o_series || (name.starts_with("gpt-5") && !name.starts_with("gpt-5-chat"))
}

/// Flatten IR content to plain text, for slots that take only a string.
fn ir_text(contents: &[Content]) -> String {
    contents
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// An OpenAI message as IR content: reasoning first, then text, then any tool
/// calls — the order the IR consumers expect a turn to read in.
fn message_to_ir(msg: &crate::Message) -> Vec<Content> {
    let mut out = Vec::new();

    if let Some(reasoning) = msg.thinking() {
        out.push(Content::Reasoning {
            text: reasoning.to_string(),
            signature: None,
        });
    }

    match &msg.content {
        Some(MessageContent::Text(text)) if !text.is_empty() => {
            out.push(Content::Text(text.clone()));
        }
        Some(MessageContent::Parts(parts)) => {
            for part in parts {
                match part {
                    ContentPart::Text { text } if !text.is_empty() => {
                        out.push(Content::Text(text.clone()));
                    }
                    ContentPart::ImageUrl { image_url } => {
                        let (media_type, data) = split_data_url(&image_url.url);
                        out.push(Content::Image { media_type, data });
                    }
                    // A refusal is the model declining; it reads as assistant
                    // text downstream, which is what every consumer wants.
                    ContentPart::Refusal { refusal } if !refusal.is_empty() => {
                        out.push(Content::Text(refusal.clone()));
                    }
                    ContentPart::Text { .. } | ContentPart::Refusal { .. } => {}
                }
            }
        }
        _ => {}
    }

    for tc in msg.tool_calls.iter().flatten() {
        out.push(Content::ToolCall {
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            input: crate::json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default())),
        });
    }

    // A cache mark on the message means "cache the prefix ending here", which
    // the IR carries as a trailing breakpoint.
    if msg.cache_control.is_some() {
        out.push(Content::CacheBreakpoint);
    }

    out
}

/// IR content as an assistant message — the inverse of [`message_to_ir`].
fn ir_to_message(contents: &[Content]) -> crate::Message {
    let tool_calls: Vec<crate::ToolCall> = contents
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall { id, name, input } => Some(crate::ToolCall {
                index: None,
                id: id.clone(),
                kind: ToolType::Function,
                function: FunctionCall {
                    name: name.clone(),
                    arguments: input.to_string(),
                },
            }),
            _ => None,
        })
        .collect();

    let reasoning_content = contents.iter().find_map(|c| match c {
        Content::Reasoning { text, .. } if !text.is_empty() => Some(text.clone()),
        _ => None,
    });

    crate::Message {
        role: crate::Role::Assistant,
        content: openai_content(contents),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        reasoning_content,
        ..Default::default()
    }
}

/// Split `data:<media-type>;base64,<data>` into its parts. A URL that isn't a
/// data URL is kept whole as the data, so a plain link survives the round trip.
fn split_data_url(url: &str) -> (String, String) {
    url.strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
        .map(|(media_type, data)| (media_type.to_string(), data.to_string()))
        .unwrap_or_else(|| (String::new(), url.to_string()))
}

impl ChatCompletionChunk {
    pub fn to_ir_events(&self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let Some(choice) = self.choices.first() else {
            if let Some(ref usage) = self.usage {
                events.push(StreamEvent::Usage(Usage::from(usage)));
            }
            return events;
        };

        if let Some(text) = choice.delta.content.as_deref()
            && !text.is_empty()
        {
            events.push(StreamEvent::TextDelta(text.to_string()));
        }

        if let Some(reasoning) = choice.delta.reasoning_content.as_deref()
            && !reasoning.is_empty()
        {
            events.push(StreamEvent::ReasoningDelta(reasoning.to_string()));
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                if let Some(ref func) = tc.function {
                    if let (Some(id), Some(name)) = (&tc.id, &func.name) {
                        events.push(StreamEvent::ToolCallStart {
                            id: id.clone(),
                            name: name.clone(),
                        });
                    }
                    if let Some(args) = &func.arguments
                        && !args.is_empty()
                    {
                        events.push(StreamEvent::ToolCallDelta(args.clone()));
                    }
                }
            }
        }

        if let Some(ref usage) = self.usage {
            events.push(StreamEvent::Usage(Usage::from(usage)));
        }

        if let Some(ref reason) = choice.finish_reason {
            let stop = match reason {
                FinishReason::Stop => StopReason::End,
                FinishReason::Length => StopReason::MaxTokens,
                FinishReason::ToolCalls => StopReason::ToolUse,
                _ => StopReason::End,
            };
            events.push(StreamEvent::Stop(stop));
        }

        events
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
                message: ir_to_message(&resp.content),
                finish_reason,
                logprobs: None,
            }],
            usage: Some(OpenAiUsage::from(&resp.usage)),
            system_fingerprint: None,
        }
    }
}

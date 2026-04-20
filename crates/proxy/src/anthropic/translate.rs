//! Translate Anthropic Messages API requests into the internal
//! `ChatCompletionRequest` shape.
//!
//! # Known lossiness
//!
//! Fields we deliberately do not preserve yet:
//! - `cache_control` markers on blocks (prompt caching opt-in).
//! - `metadata.user_id` and other top-level `metadata` fields.
//! - Thinking block `signature` (only load-bearing when replaying extended
//!   thinking back to the Anthropic upstream, which the inbound path does not
//!   do — our providers receive OpenAI-shaped reasoning_content).
//! - Non-text content inside `system` blocks (images get dropped).
//! - Non-text content inside `tool_result.content` when sent as blocks.
//! - `disable_parallel_tool_use` on `tool_choice`.

use crabllm_core::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicSystem, AnthropicTool, AnthropicUsage, ChatCompletionRequest, ChatCompletionResponse,
    Error, FinishReason, FunctionCall, FunctionDef, Message, Role, Stop, Tool, ToolCall,
    ToolChoice, ToolResultContent, ToolType, Usage,
};

pub fn to_chat_completion(req: AnthropicRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    if let Some(system) = req.system {
        let text = match system {
            AnthropicSystem::Text(s) => s,
            AnthropicSystem::Blocks(blocks) => flatten_text_blocks(&blocks),
        };
        if !text.is_empty() {
            messages.push(Message::system(text));
        }
    }

    for msg in req.messages {
        append_message(&mut messages, msg);
    }

    let stop = req.stop_sequences.map(|mut seqs| {
        if seqs.len() == 1 {
            Stop::Single(seqs.pop().unwrap())
        } else {
            Stop::Multiple(seqs)
        }
    });

    let tools = req
        .tools
        .map(|tools| tools.into_iter().map(convert_tool).collect());
    let tool_choice = req.tool_choice.as_ref().and_then(translate_tool_choice);

    ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: None,
        stream: req.stream,
        stop,
        tools,
        tool_choice,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        user: None,
        reasoning_effort: None,
        thinking: req.thinking,
        anthropic_max_tokens: Some(req.max_tokens),
        extra: serde_json::Map::new(),
    }
}

fn append_message(out: &mut Vec<Message>, msg: AnthropicMessage) {
    let is_assistant = msg.role == "assistant";

    let blocks = match msg.content {
        AnthropicContent::Text(s) => {
            out.push(if is_assistant {
                Message::assistant(s)
            } else {
                Message::user(s)
            });
            return;
        }
        AnthropicContent::Blocks(b) => b,
    };

    if is_assistant {
        out.push(assistant_from_blocks(blocks));
    } else {
        append_user_blocks(out, blocks);
    }
}

/// Collapse an assistant block list into a single `Message`.
///
/// Anthropic assistant turns may carry interleaved thinking, text, and tool_use
/// blocks. OpenAI's format has a flat `content` string + `tool_calls` array +
/// `reasoning_content`, so we concatenate text, collect tool_use into
/// tool_calls, and stash any thinking into reasoning_content.
fn assistant_from_blocks(blocks: Vec<AnthropicContentBlock>) -> Message {
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut reasoning = String::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text: t } => text.push_str(&t),
            AnthropicContentBlock::Thinking { thinking, .. } => reasoning.push_str(&thinking),
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    index: None,
                    id,
                    kind: ToolType::Function,
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                    },
                });
            }
            // Image/ToolResult are not valid on assistant turns.
            AnthropicContentBlock::Image { .. } | AnthropicContentBlock::ToolResult { .. } => {}
        }
    }

    let content = if text.is_empty() && !tool_calls.is_empty() {
        None
    } else {
        Some(serde_json::Value::String(text))
    };

    Message {
        role: Role::Assistant,
        content,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        name: None,
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        extra: serde_json::Map::new(),
    }
}

/// User turns may interleave text/image content and tool_result blocks. OpenAI
/// splits these into distinct messages: a user message for text/image parts
/// and a tool message per tool_result. Emit in original order so tool results
/// stay positioned correctly relative to any follow-up user text.
fn append_user_blocks(out: &mut Vec<Message>, blocks: Vec<AnthropicContentBlock>) {
    let mut user_parts: Vec<serde_json::Value> = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                user_parts.push(serde_json::json!({"type": "text", "text": text}));
            }
            AnthropicContentBlock::Image { source } => {
                if let Some(url) = image_source_to_url(&source) {
                    user_parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": {"url": url},
                    }));
                }
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
            } => {
                flush_user_parts(out, &mut user_parts);
                out.push(Message {
                    role: Role::Tool,
                    content: Some(serde_json::Value::String(tool_result_text(content))),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id),
                    name: None,
                    reasoning_content: None,
                    extra: serde_json::Map::new(),
                });
            }
            // Thinking/ToolUse are not valid on user turns.
            AnthropicContentBlock::Thinking { .. } | AnthropicContentBlock::ToolUse { .. } => {}
        }
    }

    flush_user_parts(out, &mut user_parts);
}

fn flush_user_parts(out: &mut Vec<Message>, parts: &mut Vec<serde_json::Value>) {
    if parts.is_empty() {
        return;
    }
    // If the only part is a text part, collapse to a plain string content so
    // the OpenAI provider sees the simpler `content: "..."` shape.
    let content =
        if parts.len() == 1 && parts[0].get("type").and_then(|t| t.as_str()) == Some("text") {
            parts[0]
                .get("text")
                .cloned()
                .unwrap_or(serde_json::Value::String(String::new()))
        } else {
            serde_json::Value::Array(std::mem::take(parts))
        };
    parts.clear();
    out.push(Message {
        role: Role::User,
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning_content: None,
        extra: serde_json::Map::new(),
    });
}

fn tool_result_text(content: ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(s) => s,
        ToolResultContent::Blocks(blocks) => flatten_text_blocks(&blocks),
    }
}

fn flatten_text_blocks(blocks: &[AnthropicContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let AnthropicContentBlock::Text { text } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

fn image_source_to_url(source: &serde_json::Value) -> Option<String> {
    let kind = source.get("type").and_then(|t| t.as_str())?;
    match kind {
        "base64" => {
            let media_type = source.get("media_type").and_then(|t| t.as_str())?;
            let data = source.get("data").and_then(|t| t.as_str())?;
            Some(format!("data:{media_type};base64,{data}"))
        }
        "url" => source
            .get("url")
            .and_then(|u| u.as_str())
            .map(str::to_string),
        _ => None,
    }
}

fn convert_tool(t: AnthropicTool) -> Tool {
    let mut schema = t.input_schema;
    crabllm_provider::schema::inline_refs(&mut schema);
    crabllm_provider::schema::strip_fields(
        &mut schema,
        &["propertyNames", "exclusiveMinimum", "exclusiveMaximum", "const"],
    );
    Tool {
        kind: ToolType::Function,
        function: FunctionDef {
            name: t.name,
            description: t.description,
            parameters: Some(schema),
        },
        strict: None,
    }
}

fn translate_tool_choice(value: &serde_json::Value) -> Option<ToolChoice> {
    let kind = value.get("type")?.as_str()?;
    match kind {
        "auto" => Some(ToolChoice::Auto),
        "any" => Some(ToolChoice::Required),
        "tool" => value
            .get("name")
            .and_then(|n| n.as_str())
            .map(|name| ToolChoice::Function {
                name: name.to_string(),
            }),
        "none" => Some(ToolChoice::Disabled),
        _ => None,
    }
}

/// Translate an internal `ChatCompletionResponse` back to the Anthropic
/// Messages API response wire shape.
///
/// Errors when the response has zero choices or missing usage — both indicate
/// a provider bug or transport failure, and papering over them with empty
/// defaults would corrupt billing and hide real problems.
pub fn from_chat_completion(resp: ChatCompletionResponse) -> Result<AnthropicResponse, Error> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("provider returned zero choices".into()))?;
    let usage = resp
        .usage
        .ok_or_else(|| Error::Internal("provider returned no usage".into()))?;

    let stop_reason = choice.finish_reason.as_ref().map(finish_reason_to_stop);
    let content = message_to_blocks(choice.message);

    Ok(AnthropicResponse {
        id: resp.id,
        r#type: "message".to_string(),
        role: "assistant".to_string(),
        model: resp.model,
        content,
        stop_reason,
        stop_sequence: None,
        usage: usage_to_anthropic(usage),
    })
}

fn message_to_blocks(msg: Message) -> Vec<AnthropicContentBlock> {
    let mut blocks = Vec::new();

    if let Some(reasoning) = msg.reasoning_content
        && !reasoning.is_empty()
    {
        blocks.push(AnthropicContentBlock::Thinking {
            thinking: reasoning,
            signature: None,
        });
    }

    if let Some(text) = msg.content.as_ref().and_then(|v| v.as_str())
        && !text.is_empty()
    {
        blocks.push(AnthropicContentBlock::Text {
            text: text.to_string(),
        });
    }

    if let Some(tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            let input = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            blocks.push(AnthropicContentBlock::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
            });
        }
    }

    // Anthropic responses always carry at least one block; emit an empty text
    // block rather than a bare empty array so SDKs don't choke.
    if blocks.is_empty() {
        blocks.push(AnthropicContentBlock::Text {
            text: String::new(),
        });
    }

    blocks
}

fn finish_reason_to_stop(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Stop => "end_turn".to_string(),
        FinishReason::Length => "max_tokens".to_string(),
        FinishReason::ToolCalls => "tool_use".to_string(),
        // Not a documented Anthropic value, but honesty beats papering over a
        // safety event. SDKs treat unknown stop_reason as a plain string.
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::Custom(s) => s.clone(),
    }
}

fn usage_to_anthropic(u: Usage) -> AnthropicUsage {
    AnthropicUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        cache_read_input_tokens: u.prompt_cache_hit_tokens,
        cache_creation_input_tokens: u.prompt_cache_miss_tokens,
    }
}

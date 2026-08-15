use crate::{
    Usage,
    ir::{self, Content, Message, Role, StopReason},
    types::gemini,
};
use std::collections::{HashMap, VecDeque};

impl From<&crate::gemini::Request> for ir::Request {
    fn from(req: &crate::gemini::Request) -> Self {
        let mut messages = Vec::with_capacity(req.contents.len());
        let mut pending_calls: HashMap<String, VecDeque<String>> = HashMap::new();
        let mut call_counter: u32 = 0;

        for content in &req.contents {
            let role = match content.role {
                Some(gemini::Role::Model) => Role::Assistant,
                _ => Role::User,
            };
            let mut blocks = Vec::with_capacity(content.parts.len());
            for part in &content.parts {
                blocks.extend(gemini_part_to_content(
                    part,
                    &mut pending_calls,
                    &mut call_counter,
                ));
            }
            if !blocks.is_empty() {
                messages.push(Message {
                    role,
                    content: blocks,
                });
            }
        }

        let system = req.system_instruction.as_ref().map(|c| {
            c.parts
                .iter()
                .filter_map(|p| p.text.as_ref().map(|t| Content::Text(t.clone())))
                .collect()
        });

        let (max_tokens, temperature, top_p, stop) = req
            .generation_config
            .as_ref()
            .map(|cfg| {
                (
                    cfg.max_output_tokens,
                    cfg.temperature,
                    cfg.top_p,
                    cfg.stop_sequences.clone(),
                )
            })
            .unwrap_or_default();

        let tools = req.tools.as_ref().map(|defs| {
            defs.iter()
                .flat_map(|d| d.function_declarations.iter())
                .map(|f| ir::Tool {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: f.parameters.clone(),
                })
                .collect()
        });

        ir::Request {
            model: String::new(),
            system,
            messages,
            max_tokens: max_tokens.unwrap_or(4096),
            temperature,
            top_p,
            stop,
            tools,
            tool_choice: None,
            thinking: None,
            stream: false,
        }
    }
}

fn gemini_part_to_content(
    part: &gemini::Part,
    pending_calls: &mut HashMap<String, VecDeque<String>>,
    call_counter: &mut u32,
) -> Vec<Content> {
    let mut out = Vec::new();
    if let Some(text) = &part.text
        && !text.is_empty()
    {
        out.push(Content::Text(text.clone()));
    }
    if let Some(fc) = &part.function_call {
        let id = format!("call_{call_counter}");
        *call_counter += 1;
        pending_calls
            .entry(fc.name.clone())
            .or_default()
            .push_back(id.clone());
        out.push(Content::ToolCall {
            id,
            name: fc.name.clone(),
            input: fc.args.clone(),
        });
    }
    if let Some(fr) = &part.function_response {
        let call_id = pending_calls
            .get_mut(&fr.name)
            .and_then(|q| q.pop_front())
            .unwrap_or_default();
        let text = match &fr.response {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out.push(Content::ToolResult {
            call_id,
            content: vec![Content::Text(text)],
        });
    }
    out
}

impl From<gemini::Response> for ir::Response {
    fn from(resp: gemini::Response) -> Self {
        let (content, stop_reason) = resp
            .candidates
            .into_iter()
            .next()
            .map(|c| {
                let stop = c.finish_reason.as_ref().map(|r| match r {
                    gemini::FinishReason::Stop => StopReason::End,
                    gemini::FinishReason::MaxTokens => StopReason::MaxTokens,
                    _ => StopReason::End,
                });
                let blocks = c
                    .content
                    .map(|content| {
                        content
                            .parts
                            .into_iter()
                            .flat_map(gemini_response_part_to_content)
                            .collect()
                    })
                    .unwrap_or_default();
                (blocks, stop)
            })
            .unwrap_or_default();

        let usage = resp
            .usage_metadata
            .as_ref()
            .map(Usage::from)
            .unwrap_or_default();

        ir::Response {
            id: String::new(),
            model: String::new(),
            content,
            stop_reason,
            usage,
        }
    }
}

fn gemini_response_part_to_content(part: gemini::Part) -> Vec<Content> {
    let mut out = Vec::new();
    if let Some(text) = part.text
        && !text.is_empty()
    {
        out.push(Content::Text(text));
    }
    if let Some(fc) = part.function_call {
        out.push(Content::ToolCall {
            id: String::new(),
            name: fc.name,
            input: fc.args,
        });
    }
    out
}

impl From<&ir::Response> for gemini::Response {
    fn from(resp: &ir::Response) -> Self {
        let mut parts = Vec::with_capacity(resp.content.len());
        for block in &resp.content {
            match block {
                Content::Text(text) if !text.is_empty() => {
                    parts.push(gemini::Part {
                        text: Some(text.clone()),
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                    });
                }
                Content::Text(_) => {}
                Content::ToolCall { name, input, .. } => {
                    parts.push(gemini::Part {
                        text: None,
                        function_call: Some(gemini::FunctionCall {
                            name: name.clone(),
                            args: input.clone(),
                        }),
                        function_response: None,
                        thought_signature: None,
                    });
                }
                _ => {}
            }
        }

        let finish_reason = resp.stop_reason.as_ref().map(|r| match r {
            StopReason::End | StopReason::ToolUse => gemini::FinishReason::Stop,
            StopReason::MaxTokens => gemini::FinishReason::MaxTokens,
        });

        gemini::Response {
            candidates: vec![gemini::Candidate {
                content: Some(gemini::Content {
                    role: Some(gemini::Role::Model),
                    parts,
                }),
                finish_reason,
            }],
            usage_metadata: Some(gemini::Usage::from(&resp.usage)),
        }
    }
}

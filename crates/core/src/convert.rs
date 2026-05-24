use crate::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicSystem, AnthropicTool, AnthropicUsage, ChatCompletionRequest, ChatCompletionResponse,
    ContentBlock, DEFAULT_MAX_TOKENS, Error, FinishReason, FunctionDef, GeminiCandidate,
    GeminiContent, GeminiFinishReason, GeminiFunctionCall, GeminiPart, GeminiRequest,
    GeminiResponse, GeminiRole, GeminiUsage, GenerationConfig, Message, Role, Stop, Tool,
    ToolChoice, ToolResultContent, ToolType, Usage,
};
use std::collections::{HashMap, VecDeque};

impl From<AnthropicRequest> for ChatCompletionRequest {
    fn from(req: AnthropicRequest) -> Self {
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
            let role = match msg.role.as_str() {
                "assistant" => Role::Assistant,
                "user" => Role::User,
                _ => Role::Custom(msg.role),
            };
            let blocks = match msg.content {
                AnthropicContent::Text(s) => vec![AnthropicContentBlock::text(s)],
                AnthropicContent::Blocks(b) => b,
            };
            messages.push(Message {
                role,
                content: blocks,
            });
        }

        let stop = req.stop_sequences.map(|mut seqs| {
            if seqs.len() == 1 {
                Stop::Single(seqs.pop().unwrap())
            } else {
                Stop::Multiple(seqs)
            }
        });

        let tools = req.tools.map(|tools| {
            tools
                .into_iter()
                .map(|t| Tool {
                    kind: ToolType::Function,
                    function: FunctionDef {
                        name: t.name,
                        description: t.description,
                        parameters: Some(t.input_schema),
                    },
                    strict: None,
                })
                .collect()
        });
        let tool_choice = req.tool_choice.as_ref().and_then(translate_tool_choice);

        ChatCompletionRequest {
            model: req.model,
            messages,
            temperature: req.temperature,
            top_p: req.top_p,
            max_tokens: Some(req.max_tokens),
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
}

impl TryFrom<ChatCompletionResponse> for AnthropicResponse {
    type Error = Error;

    fn try_from(resp: ChatCompletionResponse) -> Result<Self, Self::Error> {
        let choice = resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("provider returned zero choices".into()))?;
        let usage = resp
            .usage
            .ok_or_else(|| Error::Internal("provider returned no usage".into()))?;

        let stop_reason = choice.finish_reason.as_ref().map(finish_reason_to_stop);
        let mut content = choice.message.content;
        if content.is_empty() {
            content.push(AnthropicContentBlock::text(""));
        }

        Ok(AnthropicResponse {
            id: resp.id,
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            model: resp.model,
            content,
            stop_reason,
            stop_sequence: None,
            usage: AnthropicUsage::from(&Usage::from(&usage)),
        })
    }
}

fn flatten_text_blocks(blocks: &[AnthropicContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let AnthropicContentBlock::Text { text, .. } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
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

fn finish_reason_to_stop(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Stop => "end_turn".to_string(),
        FinishReason::Length => "max_tokens".to_string(),
        FinishReason::ToolCalls => "tool_use".to_string(),
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::Custom(s) => s.clone(),
    }
}

// ── Gemini ↔ Anthropic ──
//
// Lossy fields are dropped with `tracing::warn!`. The current lossy set on
// Gemini → Anthropic input is `thought_signature` (Gemini-thinking-model
// roundtrip marker has no Anthropic IR equivalent and is not preserved across
// the IR boundary; the GoogleProvider preserves it natively when input is
// already Gemini-shape, so this only loses signatures when the IR is
// transited through a non-Gemini provider). Future Gemini features without
// Anthropic analogues (safety settings, code-execution tool, grounding /
// search retrieval) should follow the same drop-and-warn policy when added.

impl From<&GeminiRequest> for AnthropicRequest {
    /// Convert a Gemini request to the canonical Anthropic IR.
    ///
    /// `model` is left empty — the caller (typically the proxy or the
    /// Provider trait default impl) must populate it from the URL path
    /// before forwarding.
    fn from(req: &GeminiRequest) -> Self {
        let mut messages = Vec::with_capacity(req.contents.len());
        let mut pending_calls: HashMap<String, VecDeque<String>> = HashMap::new();
        let mut call_counter: u32 = 0;

        for content in &req.contents {
            let role = match content.role {
                Some(GeminiRole::Model) => "assistant".to_string(),
                _ => "user".to_string(),
            };
            let mut blocks = Vec::with_capacity(content.parts.len());
            for part in &content.parts {
                if part.thought_signature.is_some() {
                    tracing::warn!(
                        "dropping Gemini thought_signature on Gemini→Anthropic conversion"
                    );
                }
                blocks.extend(part_to_blocks(part, &mut pending_calls, &mut call_counter));
            }
            if !blocks.is_empty() {
                messages.push(AnthropicMessage {
                    role,
                    content: AnthropicContent::Blocks(blocks),
                });
            }
        }

        let system = req.system_instruction.as_ref().map(|c| {
            let blocks: Vec<ContentBlock> = c
                .parts
                .iter()
                .filter_map(|p| p.text.as_ref().map(|t| ContentBlock::text(t.clone())))
                .collect();
            AnthropicSystem::Blocks(blocks)
        });

        let (max_tokens, temperature, top_p, stop_sequences) = req
            .generation_config
            .as_ref()
            .map(generation_config_fields)
            .unwrap_or_default();

        let tools = req.tools.as_ref().map(|defs| {
            defs.iter()
                .flat_map(|d| d.function_declarations.iter())
                .map(|f| AnthropicTool {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    input_schema: f
                        .parameters
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({})),
                })
                .collect()
        });

        AnthropicRequest {
            model: String::new(),
            messages,
            max_tokens: max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            system,
            temperature,
            top_p,
            stream: None,
            tools,
            tool_choice: None,
            stop_sequences,
            thinking: None,
        }
    }
}

fn part_to_blocks(
    part: &GeminiPart,
    pending_calls: &mut HashMap<String, VecDeque<String>>,
    call_counter: &mut u32,
) -> Vec<ContentBlock> {
    let mut out = Vec::new();
    if let Some(text) = &part.text
        && !text.is_empty()
    {
        out.push(ContentBlock::text(text.clone()));
    }
    if let Some(fc) = &part.function_call {
        let id = format!("call_{call_counter}");
        *call_counter += 1;
        pending_calls
            .entry(fc.name.clone())
            .or_default()
            .push_back(id.clone());
        out.push(ContentBlock::ToolUse {
            id,
            name: fc.name.clone(),
            input: fc.args.clone(),
            cache_control: None,
        });
    }
    if let Some(fr) = &part.function_response {
        let tool_use_id = pending_calls
            .get_mut(&fr.name)
            .and_then(|q| q.pop_front())
            .unwrap_or_default();
        let text = match &fr.response {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out.push(ContentBlock::ToolResult {
            tool_use_id,
            name: Some(fr.name.clone()),
            content: ToolResultContent::Text(text),
            cache_control: None,
        });
    }
    out
}

fn generation_config_fields(
    cfg: &GenerationConfig,
) -> (Option<u32>, Option<f64>, Option<f64>, Option<Vec<String>>) {
    (
        cfg.max_output_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.stop_sequences.clone(),
    )
}

impl TryFrom<AnthropicResponse> for GeminiResponse {
    type Error = Error;

    fn try_from(resp: AnthropicResponse) -> Result<Self, Self::Error> {
        let mut parts = Vec::with_capacity(resp.content.len());
        for block in resp.content {
            match block {
                ContentBlock::Text { text, .. } => {
                    if !text.is_empty() {
                        parts.push(GeminiPart {
                            text: Some(text),
                            function_call: None,
                            function_response: None,
                            thought_signature: None,
                        });
                    }
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    parts.push(GeminiPart {
                        text: None,
                        function_call: Some(GeminiFunctionCall { name, args: input }),
                        function_response: None,
                        thought_signature: None,
                    });
                }
                ContentBlock::ToolResult { .. } => {
                    tracing::warn!(
                        "dropping ToolResult block in Anthropic→Gemini response (assistant responses should not contain tool results)"
                    );
                }
                ContentBlock::Thinking { .. } => {
                    tracing::warn!(
                        "dropping Thinking block on Anthropic→Gemini conversion (Gemini wire format has no thinking field)"
                    );
                }
                ContentBlock::Image { .. } => {
                    tracing::warn!(
                        "dropping Image block on Anthropic→Gemini conversion (Gemini wire format on response side has no image type)"
                    );
                }
            }
        }

        let candidate = GeminiCandidate {
            content: Some(GeminiContent {
                role: Some(GeminiRole::Model),
                parts,
            }),
            finish_reason: resp.stop_reason.as_deref().map(stop_reason_to_gemini),
        };

        Ok(GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsage::from(&Usage::from(&resp.usage))),
        })
    }
}

fn stop_reason_to_gemini(reason: &str) -> GeminiFinishReason {
    match reason {
        "end_turn" | "tool_use" | "stop_sequence" => GeminiFinishReason::Stop,
        "max_tokens" => GeminiFinishReason::MaxTokens,
        "content_filter" => GeminiFinishReason::Safety,
        _ => GeminiFinishReason::Other,
    }
}

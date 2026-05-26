use crate::{
    AnthropicContent, AnthropicResponse, AnthropicSystem, AnthropicUsage, ContentBlock,
    ir::{self, Content, Message, Role, StopReason},
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

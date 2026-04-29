//! Manual smoke test for end-to-end MLX tool calling.
//!
//! Demonstrates that a tool-using request against a local MLX model
//! returns a structured `tool_calls` response, both for non-streaming
//! and streaming. Skipped from CI because there's no Apple Silicon
//! GPU. Run locally:
//!
//!   cargo run -p crabllm-mlx --example tool_capability
//!
//! Default model is `gemma-4-it-e2b-4bit`, which exercises the
//! gemma-only patches in `crates/mlx/src/gemma_patch.rs` (workaround
//! for mlx-swift-lm 3.31.3's broken GemmaFunctionParser, upstream
//! PR #215). Pass a different alias / repo id as the first arg.

use crabllm_core::{ChatCompletionRequest, FunctionDef, Message, Provider, Tool, ToolType};
use crabllm_mlx::{MlxPool, MlxProvider};
use futures::StreamExt;
use std::sync::Arc;

const DEFAULT_MODEL: &str = "gemma-4-it-e2b-4bit";

fn weather_tool() -> Tool {
    Tool {
        kind: ToolType::Function,
        function: FunctionDef {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a city.".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            })),
        },
        strict: None,
    }
}

fn tool_request(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![Message::user("What's the weather in Tokyo? Use the tool.")],
        temperature: Some(0.0),
        top_p: None,
        max_tokens: Some(256),
        stream: None,
        stop: None,
        tools: Some(vec![weather_tool()]),
        tool_choice: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        user: None,
        reasoning_effort: None,
        thinking: None,
        anthropic_max_tokens: None,
        extra: serde_json::Map::new(),
    }
}

#[tokio::main]
async fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    eprintln!("model: {model}\n");

    let pool = Arc::new(MlxPool::new(1800).expect("pool"));
    let provider = MlxProvider::new(pool);

    eprintln!("[1/2] non-streaming with tools");
    let req = tool_request(&model);
    match provider.chat_completion(&req).await {
        Ok(resp) => {
            let msg = &resp.choices[0].message;
            let text = msg.content.as_ref().and_then(|v| v.as_str()).unwrap_or("");
            let usage = resp.usage.as_ref();
            eprintln!(
                "      {} prompt / {} completion tokens",
                usage.map(|u| u.prompt_tokens).unwrap_or(0),
                usage.map(|u| u.completion_tokens).unwrap_or(0)
            );
            eprintln!("      finish_reason: {:?}", resp.choices[0].finish_reason);
            eprintln!("      tool_calls: {:?}", msg.tool_calls);
            eprintln!("      content: {text:?}\n");
        }
        Err(e) => {
            eprintln!("      err: {e}\n");
            std::process::exit(1);
        }
    }

    eprintln!("[2/2] streaming with tools");
    let req = tool_request(&model);
    match provider.chat_completion_stream(&req).await {
        Ok(mut stream) => {
            let mut content = String::new();
            let mut final_chunk = None;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(c) => {
                        if let Some(d) = c.choices.first() {
                            if let Some(text) = &d.delta.content {
                                content.push_str(text);
                            }
                            if d.finish_reason.is_some() || d.delta.tool_calls.is_some() {
                                final_chunk = Some(c);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("      stream err: {e}");
                        return;
                    }
                }
            }
            eprintln!("      content: {content:?}");
            if let Some(fc) = final_chunk {
                eprintln!("      finish_reason: {:?}", fc.choices[0].finish_reason);
                eprintln!("      tool_calls: {:?}", fc.choices[0].delta.tool_calls);
            }
        }
        Err(e) => eprintln!("      err: {e}"),
    }
}

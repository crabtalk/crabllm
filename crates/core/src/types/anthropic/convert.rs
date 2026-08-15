//! Conversions into the Anthropic tool wire shape.

use crate::types::{anthropic::Tool, openai};
use serde_json::{Value, json};

impl From<&openai::Tool> for Tool {
    fn from(tool: &openai::Tool) -> Self {
        Self {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            // Anthropic rejects a tool without an object schema, so a
            // parameterless tool still declares an empty object.
            input_schema: tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({"type": "object"})),
            cache_control: None,
        }
    }
}

/// Encode a tool choice for the Messages API.
///
/// Deliberately not `From<&openai::ToolChoice> for Value`: that conversion
/// is already taken by the OpenAI encoding (`"auto"`, `"none"`, …), and a
/// second one to the same target would resolve by reference-ness, which is
/// no way to pick a wire format.
pub fn tool_choice(choice: &openai::ToolChoice) -> Value {
    match choice {
        openai::ToolChoice::Auto => json!({"type": "auto"}),
        openai::ToolChoice::Required => json!({"type": "any"}),
        openai::ToolChoice::Function { name } => json!({"type": "tool", "name": name}),
        openai::ToolChoice::Disabled => json!({"type": "none"}),
    }
}

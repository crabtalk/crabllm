//! Manual `ToSchema` implementations for types with custom Serialize/Deserialize.
//!
//! Types that derive Serialize get `#[cfg_attr(feature = "openapi", derive(ToSchema))]`
//! directly. Types with custom impls need manual schemas here because utoipa's
//! derive macro can't infer the schema from custom serialization logic.

use utoipa::openapi::schema::{ObjectBuilder, SchemaType, Type};

use crate::{EmbeddingInput, FinishReason, ProviderKind, Role, Stop, ToolChoice};

macro_rules! string_schema {
    ($ty:ty, $desc:expr, $($variant:literal),+ $(,)?) => {
        impl utoipa::PartialSchema for $ty {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                ObjectBuilder::new()
                    .schema_type(SchemaType::new(Type::String))
                    .description(Some($desc))
                    .enum_values::<[&str; 0], &str>(None)
                    .build()
                    .into()
            }
        }
        impl utoipa::ToSchema for $ty {}
    };
}

macro_rules! any_schema {
    ($ty:ty, $desc:expr) => {
        impl utoipa::PartialSchema for $ty {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                ObjectBuilder::new().description(Some($desc)).build().into()
            }
        }
        impl utoipa::ToSchema for $ty {}
    };
}

string_schema!(
    Role,
    "One of: user, assistant, system, tool, developer",
    "user",
    "assistant",
    "system",
    "tool",
    "developer"
);
string_schema!(
    FinishReason,
    "One of: stop, length, tool_calls, content_filter",
    "stop",
    "length",
    "tool_calls",
    "content_filter"
);
any_schema!(
    ToolChoice,
    "String (\"none\", \"auto\", \"required\") or object {\"type\":\"function\",\"function\":{\"name\":\"...\"}}"
);
any_schema!(
    Stop,
    "A string or array of strings where the model should stop"
);
any_schema!(EmbeddingInput, "A string or array of strings to embed");
any_schema!(
    ProviderKind,
    "One of: openai, anthropic, google, bedrock, ollama, azure — or any self-defined name (requires base_url, dispatched as OpenAI-compatible)"
);

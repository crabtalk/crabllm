use crabllm_core::{
    AudioSpeechRequest, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, CompletionTokensDetails, Delta, Embedding, EmbeddingInput, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, FinishReason, FunctionCall, FunctionCallDelta, FunctionDef,
    ImageRequest, KeyRateLimit, Message, Model, ModelList, PricingConfig, ProviderKind, Role, Stop,
    Tool, ToolCall, ToolCallDelta, ToolChoice, ToolType, Usage,
};
use utoipa::OpenApi;
use utoipa::openapi::{
    ContentBuilder, HttpMethod, InfoBuilder, PathItem, PathsBuilder, Ref, RefOr, Required,
    ResponseBuilder, Responses, ResponsesBuilder, Tag,
    path::{OperationBuilder, ParameterBuilder, ParameterIn},
    request_body::{RequestBody, RequestBodyBuilder},
    schema::{KnownFormat, ObjectBuilder, SchemaFormat, SchemaType, Type},
    security::{HttpAuthScheme, HttpBuilder, SecurityRequirement, SecurityScheme},
};

use crate::admin::{CreateKeyRequest, KeyResponse, KeySummary};
use crate::admin_providers::{CreateProviderRequest, ProviderSummary};
use crate::ext::usage::UsageEntry;

const TAG_API: &str = "API";
const TAG_ADMIN_KEYS: &str = "Admin / Keys";
const TAG_ADMIN_PROVIDERS: &str = "Admin / Providers";
const TAG_ADMIN_USAGE: &str = "Admin / Usage";
const TAG_INFRA: &str = "Infrastructure";

#[derive(OpenApi)]
#[openapi(components(schemas(
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Message,
    ToolCall,
    FunctionCall,
    Tool,
    FunctionDef,
    ToolType,
    Choice,
    Usage,
    CompletionTokensDetails,
    ChunkChoice,
    Delta,
    ToolCallDelta,
    FunctionCallDelta,
    Role,
    FinishReason,
    ToolChoice,
    Stop,
    EmbeddingRequest,
    EmbeddingResponse,
    Embedding,
    EmbeddingUsage,
    EmbeddingInput,
    ImageRequest,
    AudioSpeechRequest,
    Model,
    ModelList,
    PricingConfig,
    KeyRateLimit,
    ProviderKind,
    CreateKeyRequest,
    KeyResponse,
    KeySummary,
    CreateProviderRequest,
    ProviderSummary,
    UsageEntry,
)))]
struct Components;

fn op(tag: &str, summary: &str) -> OperationBuilder {
    OperationBuilder::new().summary(Some(summary)).tag(tag)
}

/// Build a PathItem with multiple HTTP methods sharing one tag.
fn multi(tag: &str, ops: &[(HttpMethod, &str, Option<RequestBody>, Responses)]) -> PathItem {
    let mut item = PathItem::default();
    for (method, summary, body, responses) in ops {
        let mut builder = op(tag, summary).responses(responses.clone());
        if let Some(b) = body {
            builder = builder.request_body(Some(b.clone()));
        }
        let operation = builder.build();
        match method {
            HttpMethod::Get => item.get = Some(operation),
            HttpMethod::Post => item.post = Some(operation),
            HttpMethod::Put => item.put = Some(operation),
            HttpMethod::Patch => item.patch = Some(operation),
            HttpMethod::Delete => item.delete = Some(operation),
            _ => {}
        }
    }
    item
}

fn query(name: &str) -> ParameterBuilder {
    ParameterBuilder::new()
        .name(name)
        .parameter_in(ParameterIn::Query)
}

fn schema_ref(name: &str) -> Ref {
    Ref::from_schema_name(name)
}

fn json_body(name: &str) -> RequestBody {
    RequestBodyBuilder::new()
        .content(
            "application/json",
            ContentBuilder::new().schema(Some(schema_ref(name))).build(),
        )
        .required(Some(Required::True))
        .build()
}

fn json_merge_body() -> RequestBody {
    RequestBodyBuilder::new()
        .content(
            "application/json",
            ContentBuilder::new()
                .schema(Some(RefOr::T(
                    ObjectBuilder::new()
                        .description(Some(
                            "JSON Merge Patch — only the fields you want to change",
                        ))
                        .build()
                        .into(),
                )))
                .build(),
        )
        .required(Some(Required::True))
        .build()
}

fn json_ok(name: &str, desc: &str) -> Responses {
    ResponsesBuilder::new()
        .response(
            "200",
            ResponseBuilder::new()
                .description(desc)
                .content(
                    "application/json",
                    ContentBuilder::new().schema(Some(schema_ref(name))).build(),
                )
                .build(),
        )
        .build()
}

fn json_array_ok(name: &str, desc: &str) -> Responses {
    let array = ObjectBuilder::new()
        .schema_type(SchemaType::new(Type::Array))
        .build();
    let mut content = ContentBuilder::new()
        .schema(Some(RefOr::T(array.into())))
        .build();
    // Inline an `items` ref via raw schema. Easier: build the array schema
    // with items pointing at the named schema.
    let items_schema = utoipa::openapi::schema::ArrayBuilder::new()
        .items(schema_ref(name))
        .build();
    content.schema = Some(RefOr::T(items_schema.into()));
    ResponsesBuilder::new()
        .response(
            "200",
            ResponseBuilder::new()
                .description(desc)
                .content("application/json", content)
                .build(),
        )
        .build()
}

fn binary_ok(mime: &str, desc: &str) -> Responses {
    let bin = ObjectBuilder::new()
        .schema_type(SchemaType::new(Type::String))
        .format(Some(SchemaFormat::KnownFormat(KnownFormat::Binary)))
        .build();
    ResponsesBuilder::new()
        .response(
            "200",
            ResponseBuilder::new()
                .description(desc)
                .content(
                    mime,
                    ContentBuilder::new()
                        .schema(Some(RefOr::T(bin.into())))
                        .build(),
                )
                .build(),
        )
        .build()
}

fn no_content(desc: &str) -> Responses {
    ResponsesBuilder::new()
        .response("204", ResponseBuilder::new().description(desc).build())
        .build()
}

fn empty_ok(desc: &str) -> Responses {
    ResponsesBuilder::new()
        .response(
            "200",
            ResponseBuilder::new()
                .description(desc)
                .content(
                    "application/json",
                    ContentBuilder::new()
                        .schema(Some(RefOr::T(ObjectBuilder::new().build().into())))
                        .build(),
                )
                .build(),
        )
        .build()
}

pub fn spec() -> utoipa::openapi::OpenApi {
    let paths = PathsBuilder::new()
        .path(
            "/v1/chat/completions",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Create a chat completion")
                    .description(Some(
                        "Returns a single JSON response, or an SSE stream of \
                         `ChatCompletionChunk` events when `stream=true`.",
                    ))
                    .request_body(Some(json_body("ChatCompletionRequest")))
                    .responses(json_ok(
                        "ChatCompletionResponse",
                        "Chat completion (or SSE stream when stream=true)",
                    )),
            ),
        )
        .path(
            "/v1/messages",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Create a message (Anthropic format)").description(Some(
                    "Anthropic-style messages endpoint. Body and response follow \
                         the Anthropic Messages API; SSE is returned when stream=true.",
                )),
            ),
        )
        .path(
            "/v1/embeddings",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Create embeddings")
                    .request_body(Some(json_body("EmbeddingRequest")))
                    .responses(json_ok("EmbeddingResponse", "Embedding vectors")),
            ),
        )
        .path(
            "/v1/images/generations",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Generate images")
                    .request_body(Some(json_body("ImageRequest")))
                    .responses(binary_ok("image/png", "Generated image bytes")),
            ),
        )
        .path(
            "/v1/audio/speech",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Generate speech audio")
                    .request_body(Some(json_body("AudioSpeechRequest")))
                    .responses(binary_ok("audio/mpeg", "Synthesized audio bytes")),
            ),
        )
        .path(
            "/v1/audio/transcriptions",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Transcribe audio")
                    .description(Some(
                        "Multipart form upload: `model` field plus an audio file.",
                    ))
                    .request_body(Some(
                        RequestBodyBuilder::new()
                            .content(
                                "multipart/form-data",
                                ContentBuilder::new()
                                    .schema(Some(RefOr::T(ObjectBuilder::new().build().into())))
                                    .build(),
                            )
                            .required(Some(Required::True))
                            .build(),
                    ))
                    .responses(empty_ok("Transcription result")),
            ),
        )
        .path(
            "/v1/models",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_API, "List available models")
                    .responses(json_ok("ModelList", "Models the caller can access")),
            ),
        )
        .path(
            "/v1/usage",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_API, "Get usage for the authenticated key")
                    .parameter(query("model"))
                    .responses(json_array_ok("UsageEntry", "Usage rows for this key")),
            ),
        )
        .path(
            "/v1/admin/keys",
            multi(
                TAG_ADMIN_KEYS,
                &[
                    (
                        HttpMethod::Post,
                        "Create a virtual API key",
                        Some(json_body("CreateKeyRequest")),
                        json_ok(
                            "KeyResponse",
                            "Newly created key (full secret returned once)",
                        ),
                    ),
                    (
                        HttpMethod::Get,
                        "List all virtual keys",
                        None,
                        json_array_ok("KeySummary", "All known keys with masked secrets"),
                    ),
                ],
            ),
        )
        .path(
            "/v1/admin/keys/{name}",
            multi(
                TAG_ADMIN_KEYS,
                &[
                    (
                        HttpMethod::Get,
                        "Get key details",
                        None,
                        json_ok("KeySummary", "Key details"),
                    ),
                    (
                        HttpMethod::Patch,
                        "Update a key (models, rate_limit)",
                        Some(json_merge_body()),
                        json_ok("KeySummary", "Updated key"),
                    ),
                    (
                        HttpMethod::Delete,
                        "Revoke a virtual key",
                        None,
                        no_content("Key revoked"),
                    ),
                ],
            ),
        )
        .path(
            "/v1/admin/providers",
            multi(
                TAG_ADMIN_PROVIDERS,
                &[
                    (
                        HttpMethod::Post,
                        "Create a provider",
                        Some(json_body("CreateProviderRequest")),
                        json_ok("ProviderSummary", "Newly created provider (secrets masked)"),
                    ),
                    (
                        HttpMethod::Get,
                        "List all providers",
                        None,
                        json_array_ok("ProviderSummary", "All known providers (secrets masked)"),
                    ),
                ],
            ),
        )
        .path(
            "/v1/admin/providers/{name}",
            multi(
                TAG_ADMIN_PROVIDERS,
                &[
                    (
                        HttpMethod::Get,
                        "Get provider details",
                        None,
                        json_ok("ProviderSummary", "Provider details (secrets masked)"),
                    ),
                    (
                        HttpMethod::Patch,
                        "Update a provider",
                        Some(json_merge_body()),
                        json_ok("ProviderSummary", "Updated provider (secrets masked)"),
                    ),
                    (
                        HttpMethod::Delete,
                        "Delete a provider",
                        None,
                        no_content("Provider deleted"),
                    ),
                ],
            ),
        )
        .path(
            "/v1/admin/usage",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_ADMIN_USAGE, "Global usage view")
                    .parameter(query("name"))
                    .parameter(query("model"))
                    .responses(json_array_ok("UsageEntry", "Usage rows across all keys")),
            ),
        )
        .path(
            "/v1/admin/logs",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_ADMIN_USAGE, "Query audit logs")
                    .parameter(query("key"))
                    .parameter(query("model"))
                    .parameter(query("since"))
                    .parameter(query("until"))
                    .parameter(query("limit"))
                    .responses(empty_ok("Audit log entries")),
            ),
        )
        .path(
            "/v1/budget",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_ADMIN_USAGE, "Get budget status per key")
                    .responses(empty_ok("Budget status per key")),
            ),
        )
        .path(
            "/v1/cache",
            PathItem::new(
                HttpMethod::Delete,
                op(TAG_ADMIN_USAGE, "Clear response cache").responses(no_content("Cache cleared")),
            ),
        )
        .path(
            "/health",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_INFRA, "Health check").responses(empty_ok("Service healthy")),
            ),
        )
        .path(
            "/metrics",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_INFRA, "Prometheus metrics").responses(binary_ok(
                    "text/plain; version=0.0.4",
                    "Prometheus exposition format",
                )),
            ),
        )
        .build();

    let tags = vec![
        Tag::new(TAG_API),
        Tag::new(TAG_ADMIN_KEYS),
        Tag::new(TAG_ADMIN_PROVIDERS),
        Tag::new(TAG_ADMIN_USAGE),
        Tag::new(TAG_INFRA),
    ];

    let mut doc = <Components as OpenApi>::openapi();
    doc.info = InfoBuilder::new()
        .title("CrabLLM API")
        .version(env!("CARGO_PKG_VERSION"))
        .description(Some("High-performance LLM API gateway"))
        .build();
    doc.paths = paths;
    doc.tags = Some(tags);
    doc.security = Some(vec![SecurityRequirement::new(
        "BearerAuth",
        Vec::<String>::new(),
    )]);

    let components = doc
        .components
        .get_or_insert_with(utoipa::openapi::Components::default);
    components.security_schemes.insert(
        "BearerAuth".to_string(),
        SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
    );

    doc
}

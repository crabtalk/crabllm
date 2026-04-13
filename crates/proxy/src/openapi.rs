use utoipa::openapi::{
    ComponentsBuilder, HttpMethod, InfoBuilder, PathItem, PathsBuilder,
    path::{OperationBuilder, ParameterBuilder, ParameterIn},
    security::{HttpAuthScheme, HttpBuilder, SecurityRequirement, SecurityScheme},
};

fn op(summary: &str) -> OperationBuilder {
    OperationBuilder::new().summary(Some(summary))
}

/// Build a PathItem with multiple HTTP methods.
fn multi(ops: &[(HttpMethod, &str)]) -> PathItem {
    let mut item = PathItem::default();
    for (method, summary) in ops {
        let operation = op(summary).build();
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

pub fn spec() -> utoipa::openapi::OpenApi {
    let paths = PathsBuilder::new()
        // ── User-facing API ──
        .path(
            "/v1/chat/completions",
            PathItem::new(HttpMethod::Post, op("Create a chat completion")),
        )
        .path(
            "/v1/messages",
            PathItem::new(HttpMethod::Post, op("Create a message (Anthropic format)")),
        )
        .path(
            "/v1/embeddings",
            PathItem::new(HttpMethod::Post, op("Create embeddings")),
        )
        .path(
            "/v1/images/generations",
            PathItem::new(HttpMethod::Post, op("Generate images")),
        )
        .path(
            "/v1/audio/speech",
            PathItem::new(HttpMethod::Post, op("Generate speech audio")),
        )
        .path(
            "/v1/audio/transcriptions",
            PathItem::new(HttpMethod::Post, op("Transcribe audio")),
        )
        .path(
            "/v1/models",
            PathItem::new(HttpMethod::Get, op("List available models")),
        )
        .path(
            "/v1/usage",
            PathItem::new(
                HttpMethod::Get,
                op("Get usage for the authenticated key").parameter(query("model")),
            ),
        )
        // ── Admin — keys ──
        .path(
            "/v1/admin/keys",
            multi(&[
                (HttpMethod::Post, "Create a virtual API key"),
                (HttpMethod::Get, "List all virtual keys"),
            ]),
        )
        .path(
            "/v1/admin/keys/{name}",
            multi(&[
                (HttpMethod::Get, "Get key details"),
                (HttpMethod::Patch, "Update a key (models, rate_limit)"),
                (HttpMethod::Delete, "Revoke a virtual key"),
            ]),
        )
        // ── Admin — models ──
        .path(
            "/v1/admin/models",
            PathItem::new(HttpMethod::Get, op("List model metadata")),
        )
        .path(
            "/v1/admin/models/flush",
            PathItem::new(HttpMethod::Post, op("Flush model overrides to config")),
        )
        .path(
            "/v1/admin/models/{model}",
            multi(&[
                (HttpMethod::Get, "Get model metadata"),
                (HttpMethod::Put, "Upsert model metadata"),
                (HttpMethod::Delete, "Remove model override"),
            ]),
        )
        // ── Admin — providers ──
        .path(
            "/v1/admin/providers/reload",
            PathItem::new(HttpMethod::Post, op("Reload provider registry from config")),
        )
        // ── Admin — usage, logs, budget, cache ──
        .path(
            "/v1/admin/usage",
            PathItem::new(
                HttpMethod::Get,
                op("Global usage view")
                    .parameter(query("name"))
                    .parameter(query("model")),
            ),
        )
        .path(
            "/v1/admin/logs",
            PathItem::new(
                HttpMethod::Get,
                op("Query audit logs")
                    .parameter(query("key"))
                    .parameter(query("model"))
                    .parameter(query("since"))
                    .parameter(query("until"))
                    .parameter(query("limit")),
            ),
        )
        .path(
            "/v1/budget",
            PathItem::new(HttpMethod::Get, op("Get budget status per key")),
        )
        .path(
            "/v1/cache",
            PathItem::new(HttpMethod::Delete, op("Clear response cache")),
        )
        // ── Infrastructure ──
        .path(
            "/health",
            PathItem::new(HttpMethod::Get, op("Health check")),
        )
        .path(
            "/metrics",
            PathItem::new(HttpMethod::Get, op("Prometheus metrics")),
        )
        .build();

    utoipa::openapi::OpenApiBuilder::new()
        .info(
            InfoBuilder::new()
                .title("CrabLLM API")
                .version(env!("CARGO_PKG_VERSION"))
                .description(Some("High-performance LLM API gateway"))
                .build(),
        )
        .paths(paths)
        .security(Some(vec![SecurityRequirement::new(
            "BearerAuth",
            Vec::<String>::new(),
        )]))
        .components(Some(
            ComponentsBuilder::new()
                .security_scheme(
                    "BearerAuth",
                    SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
                )
                .build(),
        ))
        .build()
}

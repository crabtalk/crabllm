use crate::PREFIX_USAGE;
use axum::{Json, Router, extract::Query, routing::get};
use crabllm_core::{
    BoxFuture, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, RequestContext,
    Storage, storage_key,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct UsageTracker {
    storage: Arc<dyn Storage>,
}

impl UsageTracker {
    pub fn new(_config: &serde_json::Value, storage: Arc<dyn Storage>) -> Result<Self, String> {
        Ok(Self { storage })
    }

    /// Admin usage routes: `GET /v1/admin/usage?name=&model=` — global view.
    pub fn admin_routes(&self) -> Router {
        let storage = self.storage.clone();
        Router::new().route(
            "/v1/admin/usage",
            get(move |query: Query<AdminUsageQuery>| {
                let storage = storage.clone();
                async move { admin_usage_handler(storage, query.0).await }
            }),
        )
    }

    /// Record token usage for a given key and model.
    async fn record(
        &self,
        key_name: &str,
        model: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) {
        let prompt_suffix = format!("{key_name}:{model}:p");
        let completion_suffix = format!("{key_name}:{model}:c");

        let _ = self
            .storage
            .increment(
                &storage_key(&PREFIX_USAGE, prompt_suffix.as_bytes()),
                prompt_tokens as i64,
            )
            .await;
        let _ = self
            .storage
            .increment(
                &storage_key(&PREFIX_USAGE, completion_suffix.as_bytes()),
                completion_tokens as i64,
            )
            .await;
    }
}

impl crabllm_core::Extension for UsageTracker {
    fn name(&self) -> &str {
        "usage"
    }

    fn prefix(&self) -> crabllm_core::Prefix {
        PREFIX_USAGE
    }

    fn on_response(
        &self,
        ctx: &RequestContext,
        _request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> BoxFuture<'_, ()> {
        let key_name = ctx
            .key_name
            .clone()
            .unwrap_or_else(|| "__global".to_string());
        let model = ctx.model.clone();
        let usage = response.usage.clone();

        Box::pin(async move {
            if let Some(u) = usage {
                self.record(&key_name, &model, u.prompt_tokens, u.completion_tokens)
                    .await;
            }
        })
    }

    fn on_chunk(&self, ctx: &RequestContext, chunk: &ChatCompletionChunk) -> BoxFuture<'_, ()> {
        let key_name = ctx
            .key_name
            .clone()
            .unwrap_or_else(|| "__global".to_string());
        let model = ctx.model.clone();
        let usage = chunk.usage.clone();

        Box::pin(async move {
            if let Some(u) = usage {
                self.record(&key_name, &model, u.prompt_tokens, u.completion_tokens)
                    .await;
            }
        })
    }
}

#[derive(Deserialize)]
struct AdminUsageQuery {
    name: Option<String>,
    model: Option<String>,
}

#[derive(Deserialize)]
pub struct UserUsageQuery {
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct UsageEntry {
    name: String,
    model: String,
    prompt_tokens: i64,
    completion_tokens: i64,
}

/// Query usage data from storage, filtered by optional key name and model.
pub async fn query_usage(
    storage: &dyn Storage,
    name_filter: Option<&str>,
    model_filter: Option<&str>,
) -> Json<Vec<UsageEntry>> {
    let pairs = storage.list(&PREFIX_USAGE).await.unwrap_or_default();

    let mut entries: std::collections::HashMap<(String, String), (i64, i64)> =
        std::collections::HashMap::new();

    for (raw_key, raw_value) in &pairs {
        let suffix = match std::str::from_utf8(&raw_key[crabllm_core::PREFIX_LEN..]) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // suffix format: "{key_name}:{model}:{p|c}"
        let Some((rest, kind)) = suffix.rsplit_once(':') else {
            continue;
        };
        let Some((key_name, model)) = rest.split_once(':') else {
            continue;
        };

        if let Some(filter) = name_filter
            && key_name != filter
        {
            continue;
        }
        if let Some(filter) = model_filter
            && model != filter
        {
            continue;
        }

        let val = raw_value
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .map(i64::from_le_bytes)
            .unwrap_or(0);

        let entry = entries
            .entry((key_name.to_string(), model.to_string()))
            .or_insert((0, 0));

        match kind {
            "p" => entry.0 = val,
            "c" => entry.1 = val,
            _ => {}
        }
    }

    let result: Vec<UsageEntry> = entries
        .into_iter()
        .map(|((name, model), (prompt, completion))| UsageEntry {
            name,
            model,
            prompt_tokens: prompt,
            completion_tokens: completion,
        })
        .collect();

    Json(result)
}

async fn admin_usage_handler(
    storage: Arc<dyn Storage>,
    query: AdminUsageQuery,
) -> Json<Vec<UsageEntry>> {
    query_usage(&*storage, query.name.as_deref(), query.model.as_deref()).await
}

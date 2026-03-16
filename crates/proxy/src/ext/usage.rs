use axum::{Json, Router, routing::get};
use crabtalk_core::{
    BoxFuture, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Prefix,
    RequestContext, Storage, storage_key,
};
use serde::Serialize;
use std::sync::Arc;

pub struct UsageTracker {
    storage: Arc<dyn Storage>,
}

impl UsageTracker {
    pub fn new(_config: &toml::Value, storage: Arc<dyn Storage>) -> Result<Self, String> {
        Ok(Self { storage })
    }

    /// The fixed prefix for this extension.
    const PREFIX: Prefix = *b"usge";

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
                &storage_key(&Self::PREFIX, prompt_suffix.as_bytes()),
                prompt_tokens as i64,
            )
            .await;
        let _ = self
            .storage
            .increment(
                &storage_key(&Self::PREFIX, completion_suffix.as_bytes()),
                completion_tokens as i64,
            )
            .await;
    }
}

impl crabtalk_core::Extension for UsageTracker {
    fn name(&self) -> &str {
        "usage"
    }

    fn prefix(&self) -> Prefix {
        Self::PREFIX
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

    fn routes(&self) -> Option<Router> {
        let storage = self.storage.clone();
        let prefix = Self::PREFIX;
        let router = Router::new().route(
            "/v1/usage",
            get(move || {
                let storage = storage.clone();
                async move { usage_handler(storage, prefix).await }
            }),
        );
        Some(router)
    }
}

#[derive(Serialize)]
struct UsageEntry {
    key: String,
    model: String,
    prompt_tokens: i64,
    completion_tokens: i64,
}

async fn usage_handler(storage: Arc<dyn Storage>, prefix: Prefix) -> Json<Vec<UsageEntry>> {
    let pairs = storage.list(&prefix).await.unwrap_or_default();

    // Group by (key, model) — keys are PREFIX + "{key_name}:{model}:{p|c}"
    let mut entries: std::collections::HashMap<(String, String), (i64, i64)> =
        std::collections::HashMap::new();

    for (raw_key, _value) in &pairs {
        // Skip the prefix bytes, parse the suffix as UTF-8.
        let suffix = match std::str::from_utf8(&raw_key[crabtalk_core::PREFIX_LEN..]) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // suffix format: "{key_name}:{model}:{p|c}"
        // Split from the right to handle key names or models containing ":"
        let Some((rest, kind)) = suffix.rsplit_once(':') else {
            continue;
        };
        let Some((key_name, model)) = rest.split_once(':') else {
            continue;
        };

        let counter_key = storage_key(&prefix, suffix.as_bytes());
        // Read the counter value from the counters map via increment(0).
        let val = storage.increment(&counter_key, 0).await.unwrap_or(0);

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
        .map(|((key, model), (prompt, completion))| UsageEntry {
            key,
            model,
            prompt_tokens: prompt,
            completion_tokens: completion,
        })
        .collect();

    Json(result)
}

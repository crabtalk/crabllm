use crate::PREFIX_BUDGET;
use axum::{Json, Router, routing::get};
use crabllm_core::{
    BoxFuture, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ExtensionError,
    ModelInfo, RequestContext, Storage, resolve_model_info_full, storage_key,
};
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};

pub struct Budget {
    storage: Arc<dyn Storage>,
    models: HashMap<String, ModelInfo>,
    model_overrides: Arc<std::sync::RwLock<HashMap<String, ModelInfo>>>,
    default_budget_micros: i64,
    key_budgets: HashMap<String, i64>,
}

impl Budget {
    pub fn new(
        config: &serde_json::Value,
        storage: Arc<dyn Storage>,
        models: HashMap<String, ModelInfo>,
        model_overrides: Arc<std::sync::RwLock<HashMap<String, ModelInfo>>>,
    ) -> Result<Self, String> {
        let default_budget = config
            .get("default_budget")
            .and_then(|v| v.as_f64())
            .ok_or("budget: missing or invalid 'default_budget' (USD float)")?;

        if default_budget <= 0.0 {
            return Err("budget: 'default_budget' must be positive".to_string());
        }

        let default_budget_micros = (default_budget * 1_000_000.0) as i64;

        let mut key_budgets = HashMap::new();
        if let Some(keys_table) = config.get("keys").and_then(|v| v.as_object()) {
            for (key_name, key_config) in keys_table {
                let budget = key_config
                    .get("budget")
                    .and_then(|v| v.as_f64())
                    .ok_or(format!(
                        "budget: key '{key_name}' missing or invalid 'budget'"
                    ))?;
                key_budgets.insert(key_name.clone(), (budget * 1_000_000.0) as i64);
            }
        }

        Ok(Self {
            storage,
            models,
            model_overrides,
            default_budget_micros,
            key_budgets,
        })
    }

    fn budget_for_key(&self, key_name: &str) -> i64 {
        self.key_budgets
            .get(key_name)
            .copied()
            .unwrap_or(self.default_budget_micros)
    }

    fn cost_micros(
        &self,
        model: &str,
        provider: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> i64 {
        let guard = self
            .model_overrides
            .read()
            .unwrap_or_else(|e| e.into_inner());
        // Try provider-qualified key first (e.g. "openai/gpt-4o"), fall back to bare model name.
        let qualified = format!("{provider}/{model}");
        let info = resolve_model_info_full(&qualified, &guard, &self.models)
            .or_else(|| resolve_model_info_full(model, &guard, &self.models));
        let Some(info) = info else {
            return 0;
        };
        (info.cost(prompt_tokens, completion_tokens) * 1_000_000.0).round() as i64
    }

    pub fn admin_routes(&self) -> Router {
        let storage = self.storage.clone();
        let prefix = PREFIX_BUDGET;
        let default_budget = self.default_budget_micros;
        let key_budgets = self.key_budgets.clone();

        Router::new().route(
            "/v1/budget",
            get(move || {
                let storage = storage.clone();
                let key_budgets = key_budgets.clone();
                async move { budget_handler(storage, prefix, default_budget, key_budgets).await }
            }),
        )
    }

    async fn record_cost(
        &self,
        key_name: &str,
        model: &str,
        provider: &str,
        prompt: u32,
        completion: u32,
    ) {
        let micros = self.cost_micros(model, provider, prompt, completion);
        if micros > 0 {
            let key = storage_key(&PREFIX_BUDGET, key_name.as_bytes());
            let _ = self.storage.increment(&key, micros).await;
        }
    }
}

impl crabllm_core::Extension for Budget {
    fn name(&self) -> &str {
        "budget"
    }

    fn prefix(&self) -> crabllm_core::Prefix {
        PREFIX_BUDGET
    }

    fn on_request(&self, ctx: &RequestContext) -> BoxFuture<'_, Result<(), ExtensionError>> {
        let key_name = ctx
            .key_name
            .clone()
            .unwrap_or_else(|| "__global".to_string());
        let budget = self.budget_for_key(&key_name);

        Box::pin(async move {
            let key = storage_key(&PREFIX_BUDGET, key_name.as_bytes());
            let spent = self.storage.increment(&key, 0).await.unwrap_or(0);

            if spent >= budget {
                return Err(ExtensionError::new(
                    429,
                    "budget exceeded",
                    "budget_exceeded",
                ));
            }

            Ok(())
        })
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
        let provider = ctx.provider.clone();
        let usage = response.usage.clone();

        Box::pin(async move {
            if let Some(u) = usage {
                self.record_cost(
                    &key_name,
                    &model,
                    &provider,
                    u.prompt_tokens,
                    u.completion_tokens,
                )
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
        let provider = ctx.provider.clone();
        let usage = chunk.usage.clone();

        Box::pin(async move {
            if let Some(u) = usage {
                self.record_cost(
                    &key_name,
                    &model,
                    &provider,
                    u.prompt_tokens,
                    u.completion_tokens,
                )
                .await;
            }
        })
    }
}

#[derive(Serialize)]
struct BudgetEntry {
    key: String,
    spent_usd: f64,
    budget_usd: f64,
    remaining_usd: f64,
}

async fn budget_handler(
    storage: Arc<dyn Storage>,
    prefix: crabllm_core::Prefix,
    default_budget_micros: i64,
    key_budgets: HashMap<String, i64>,
) -> Json<Vec<BudgetEntry>> {
    let pairs = storage.list(&prefix).await.unwrap_or_default();

    let mut entries = Vec::new();
    for (raw_key, raw_value) in &pairs {
        let suffix = match std::str::from_utf8(&raw_key[crabllm_core::PREFIX_LEN..]) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Parse the counter value directly from the list() result bytes.
        let spent_micros = raw_value
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .map(i64::from_le_bytes)
            .unwrap_or(0);

        let budget_micros = key_budgets
            .get(suffix)
            .copied()
            .unwrap_or(default_budget_micros);

        let spent_usd = spent_micros as f64 / 1_000_000.0;
        let budget_usd = budget_micros as f64 / 1_000_000.0;

        entries.push(BudgetEntry {
            key: suffix.to_string(),
            spent_usd,
            budget_usd,
            remaining_usd: (budget_usd - spent_usd).max(0.0),
        });
    }

    Json(entries)
}

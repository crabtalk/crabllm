use crate::PREFIX_AUDIT;
use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use crabllm_core::{
    ApiError, BoxFuture, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Error,
    ModelInfo, RequestContext, Storage, storage_key,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::SystemTime};

pub struct AuditLogger {
    storage: Arc<dyn Storage>,
    models: HashMap<String, ModelInfo>,
}

impl AuditLogger {
    pub fn new(
        _config: &serde_json::Value,
        storage: Arc<dyn Storage>,
        models: HashMap<String, ModelInfo>,
    ) -> Result<Self, String> {
        Ok(Self { storage, models })
    }

    pub fn admin_routes(&self) -> Router {
        Router::new()
            .route("/v1/admin/logs", get(logs_handler))
            .with_state(self.storage.clone())
    }

    fn cost_micros(&self, model: &str, provider: &str, prompt: u32, completion: u32) -> i64 {
        let qualified = format!("{provider}/{model}");
        self.models
            .get(qualified.as_str())
            .or_else(|| self.models.get(model))
            .map(|info| (info.cost(prompt, completion) * 1_000_000.0).round() as i64)
            .unwrap_or(0)
    }

    fn write_record(&self, record: AuditRecord) {
        let ts_bytes = record.timestamp.to_be_bytes();
        let mut suffix = Vec::with_capacity(8 + record.request_id.len());
        suffix.extend_from_slice(&ts_bytes);
        suffix.extend_from_slice(record.request_id.as_bytes());
        let key = storage_key(&PREFIX_AUDIT, &suffix);

        let storage = self.storage.clone();
        // Fire-and-forget — audit logging must not block the response path.
        tokio::spawn(async move {
            match serde_json::to_vec(&record) {
                Ok(value) => {
                    if let Err(e) = storage.set(&key, value).await {
                        tracing::warn!("audit: failed to write record: {e}");
                    }
                }
                Err(e) => tracing::warn!("audit: failed to serialize record: {e}"),
            }
        });
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn error_status(e: &Error) -> u16 {
    match e {
        Error::Provider { status, .. } => *status,
        Error::Timeout => 504,
        _ => 500,
    }
}

impl crabllm_core::Extension for AuditLogger {
    fn name(&self) -> &str {
        "audit"
    }

    fn prefix(&self) -> crabllm_core::Prefix {
        PREFIX_AUDIT
    }

    fn on_response(
        &self,
        ctx: &RequestContext,
        _request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> BoxFuture<'_, ()> {
        let (prompt, completion) = response
            .usage
            .as_ref()
            .map(|u| (Some(u.prompt_tokens), Some(u.completion_tokens)))
            .unwrap_or((None, None));

        let cost_micros = match (prompt, completion) {
            (Some(p), Some(c)) => self.cost_micros(&ctx.model, &ctx.provider, p, c),
            _ => 0,
        };

        self.write_record(AuditRecord {
            request_id: ctx.request_id.clone(),
            timestamp: now_millis(),
            key_name: ctx.key_name.clone().unwrap_or_default(),
            model: ctx.model.clone(),
            provider: ctx.provider.clone(),
            prompt_tokens: prompt,
            completion_tokens: completion,
            cost_micros,
            latency_ms: ctx.started_at.elapsed().as_millis() as u64,
            status: 200,
            error: None,
        });

        Box::pin(async {})
    }

    fn on_chunk(&self, ctx: &RequestContext, chunk: &ChatCompletionChunk) -> BoxFuture<'_, ()> {
        // Record once when the final streaming chunk carries usage data.
        // Limitation: providers that don't include usage in stream chunks
        // (despite stream_options.include_usage) produce no audit record.
        // The Extension trait has no on_stream_end hook to catch this.
        if let Some(ref usage) = chunk.usage {
            let cost_micros = self.cost_micros(
                &ctx.model,
                &ctx.provider,
                usage.prompt_tokens,
                usage.completion_tokens,
            );

            self.write_record(AuditRecord {
                request_id: ctx.request_id.clone(),
                timestamp: now_millis(),
                key_name: ctx.key_name.clone().unwrap_or_default(),
                model: ctx.model.clone(),
                provider: ctx.provider.clone(),
                prompt_tokens: Some(usage.prompt_tokens),
                completion_tokens: Some(usage.completion_tokens),
                cost_micros,
                latency_ms: ctx.started_at.elapsed().as_millis() as u64,
                status: 200,
                error: None,
            });
        }

        Box::pin(async {})
    }

    fn on_error(&self, ctx: &RequestContext, error: &Error) -> BoxFuture<'_, ()> {
        self.write_record(AuditRecord {
            request_id: ctx.request_id.clone(),
            timestamp: now_millis(),
            key_name: ctx.key_name.clone().unwrap_or_default(),
            model: ctx.model.clone(),
            provider: ctx.provider.clone(),
            prompt_tokens: None,
            completion_tokens: None,
            cost_micros: 0,
            latency_ms: ctx.started_at.elapsed().as_millis() as u64,
            status: error_status(error),
            error: Some(error.to_string()),
        });

        Box::pin(async {})
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub request_id: String,
    pub timestamp: i64,
    pub key_name: String,
    pub model: String,
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    pub cost_micros: i64,
    pub latency_ms: u64,
    pub status: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Deserialize)]
struct LogQuery {
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    since: Option<i64>,
    #[serde(default)]
    until: Option<i64>,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    100
}

/// GET /v1/admin/logs — query audit log records.
///
/// Loads all records from storage and filters in memory. Acceptable for
/// moderate volumes; high-throughput deployments should migrate to a
/// dedicated time-series store.
async fn logs_handler(
    State(storage): State<Arc<dyn Storage>>,
    Query(query): Query<LogQuery>,
) -> Response {
    let pairs = match storage.list(&PREFIX_AUDIT).await {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new(e.to_string(), "server_error")),
            )
                .into_response();
        }
    };

    let mut records: Vec<AuditRecord> = pairs
        .into_iter()
        .filter_map(|(_k, v)| serde_json::from_slice(&v).ok())
        .filter(|r: &AuditRecord| {
            if let Some(ref key) = query.key
                && &r.key_name != key
            {
                return false;
            }
            if let Some(ref model) = query.model
                && &r.model != model
            {
                return false;
            }
            if let Some(since) = query.since
                && r.timestamp < since
            {
                return false;
            }
            if let Some(until) = query.until
                && r.timestamp > until
            {
                return false;
            }
            true
        })
        .collect();

    // Newest first.
    records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    records.truncate(query.limit);

    Json(records).into_response()
}

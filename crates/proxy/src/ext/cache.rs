use crate::PREFIX_CACHE;
use axum::{Router, http::StatusCode, routing::delete};
use crabllm_core::{BoxFuture, RequestContext, Storage, storage_key};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub struct Cache {
    storage: Arc<dyn Storage>,
    ttl_seconds: u64,
}

impl Cache {
    pub fn new(config: &serde_json::Value, storage: Arc<dyn Storage>) -> Result<Self, String> {
        let ttl_seconds = config
            .get("ttl_seconds")
            .and_then(|v| v.as_i64())
            .unwrap_or(300) as u64;

        Ok(Self {
            storage,
            ttl_seconds,
        })
    }

    fn cache_key(raw_request: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(raw_request);
        storage_key(&PREFIX_CACHE, &hasher.finalize())
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    pub fn admin_routes(&self) -> Router {
        let storage = self.storage.clone();
        let prefix = PREFIX_CACHE;
        Router::new().route(
            "/v1/cache",
            delete(move || {
                let storage = storage.clone();
                async move {
                    let pairs = storage.list(&prefix).await.unwrap_or_default();
                    for (key, _) in pairs {
                        let _ = storage.delete(&key).await;
                    }
                    StatusCode::NO_CONTENT
                }
            }),
        )
    }
}

impl crabllm_core::Extension for Cache {
    fn name(&self) -> &str {
        "cache"
    }

    fn prefix(&self) -> crabllm_core::Prefix {
        PREFIX_CACHE
    }

    fn on_cache_lookup(&self, raw_request: &[u8]) -> BoxFuture<'_, Option<Vec<u8>>> {
        let key = Self::cache_key(raw_request);
        let ttl = self.ttl_seconds;

        Box::pin(async move {
            let data = self.storage.get(&key).await.ok()??;
            if data.len() < 8 {
                return None;
            }

            let timestamp = u64::from_be_bytes(data[..8].try_into().ok()?);
            if Self::now_secs().saturating_sub(timestamp) > ttl {
                let _ = self.storage.delete(&key).await;
                return None;
            }

            Some(data[8..].to_vec())
        })
    }

    fn on_response(
        &self,
        ctx: &RequestContext,
        raw_request: &[u8],
        raw_response: &[u8],
    ) -> BoxFuture<'_, ()> {
        if ctx.is_stream {
            return Box::pin(async {});
        }

        let key = Self::cache_key(raw_request);
        let mut value = Vec::with_capacity(8 + raw_response.len());
        value.extend_from_slice(&Self::now_secs().to_be_bytes());
        value.extend_from_slice(raw_response);

        Box::pin(async move {
            let _ = self.storage.set(&key, value).await;
        })
    }
}

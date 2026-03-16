use axum::{Router, http::StatusCode, routing::delete};
use crabtalk_core::{
    BoxFuture, ChatCompletionRequest, ChatCompletionResponse, Prefix, RequestContext, Storage,
    storage_key,
};
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
    const PREFIX: Prefix = *b"cach";

    pub fn new(config: &toml::Value, storage: Arc<dyn Storage>) -> Result<Self, String> {
        let ttl_seconds = config
            .get("ttl_seconds")
            .and_then(|v| v.as_integer())
            .unwrap_or(300) as u64;

        Ok(Self {
            storage,
            ttl_seconds,
        })
    }

    fn cache_key(request: &ChatCompletionRequest) -> Vec<u8> {
        let json = serde_json::to_string(request).unwrap_or_default();
        let hash = Sha256::digest(json.as_bytes());
        storage_key(&Self::PREFIX, &hash)
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl crabtalk_core::Extension for Cache {
    fn name(&self) -> &str {
        "cache"
    }

    fn prefix(&self) -> Prefix {
        Self::PREFIX
    }

    fn on_cache_lookup(
        &self,
        request: &ChatCompletionRequest,
    ) -> BoxFuture<'_, Option<ChatCompletionResponse>> {
        let key = Self::cache_key(request);
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

            serde_json::from_slice(&data[8..]).ok()
        })
    }

    fn on_response(
        &self,
        ctx: &RequestContext,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> BoxFuture<'_, ()> {
        if ctx.is_stream {
            return Box::pin(async {});
        }

        let key = Self::cache_key(request);
        let Ok(json) = serde_json::to_vec(response) else {
            return Box::pin(async {});
        };

        let mut value = Vec::with_capacity(8 + json.len());
        value.extend_from_slice(&Self::now_secs().to_be_bytes());
        value.extend_from_slice(&json);

        Box::pin(async move {
            let _ = self.storage.set(&key, value).await;
        })
    }

    fn routes(&self) -> Option<Router> {
        let storage = self.storage.clone();
        let prefix = Self::PREFIX;
        let router = Router::new().route(
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
        );
        Some(router)
    }
}

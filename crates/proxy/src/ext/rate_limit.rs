use crabtalk_core::{BoxFuture, ExtensionError, Prefix, RequestContext, Storage};
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub struct RateLimit {
    storage: Arc<dyn Storage>,
    requests_per_minute: u64,
}

impl RateLimit {
    pub fn new(config: &toml::Value, storage: Arc<dyn Storage>) -> Result<Self, String> {
        let rpm = config
            .get("requests_per_minute")
            .and_then(|v| v.as_integer())
            .ok_or("rate_limit: missing or invalid 'requests_per_minute'")?;

        if rpm <= 0 {
            return Err("rate_limit: 'requests_per_minute' must be positive".to_string());
        }

        Ok(Self {
            storage,
            requests_per_minute: rpm as u64,
        })
    }
}

impl crabtalk_core::Extension for RateLimit {
    fn name(&self) -> &str {
        "rate_limit"
    }

    fn prefix(&self) -> Prefix {
        *b"rlim"
    }

    fn on_request(&self, ctx: &RequestContext) -> BoxFuture<'_, Result<(), ExtensionError>> {
        let key_name = ctx.key_name.as_deref().unwrap_or("__global");
        let minute = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 60;
        let suffix = format!("{key_name}:{minute}");
        let key = self.storage_key(suffix.as_bytes());
        let limit = self.requests_per_minute;

        Box::pin(async move {
            let count = self
                .storage
                .increment(&key, 1)
                .await
                .map_err(|e| ExtensionError::new(500, e.to_string(), "server_error"))?;

            if count as u64 > limit {
                return Err(ExtensionError::new(
                    429,
                    "rate limit exceeded",
                    "rate_limit_error",
                ));
            }

            Ok(())
        })
    }
}

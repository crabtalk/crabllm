use crabtalk_core::{
    BoxFuture, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ExtensionError,
    Prefix, RequestContext, Storage,
};
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub struct RateLimit {
    storage: Arc<dyn Storage>,
    requests_per_minute: u64,
    tokens_per_minute: Option<u64>,
}

impl RateLimit {
    pub fn new(config: &serde_json::Value, storage: Arc<dyn Storage>) -> Result<Self, String> {
        let rpm = config
            .get("requests_per_minute")
            .and_then(|v| v.as_i64())
            .ok_or("rate_limit: missing or invalid 'requests_per_minute'")?;

        if rpm <= 0 {
            return Err("rate_limit: 'requests_per_minute' must be positive".to_string());
        }

        let tpm = config
            .get("tokens_per_minute")
            .and_then(|v| v.as_i64())
            .map(|v| {
                if v <= 0 {
                    Err("rate_limit: 'tokens_per_minute' must be positive".to_string())
                } else {
                    Ok(v as u64)
                }
            })
            .transpose()?;

        Ok(Self {
            storage,
            requests_per_minute: rpm as u64,
            tokens_per_minute: tpm,
        })
    }
}

fn current_minute() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / 60
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
        let minute = current_minute();

        let rpm_suffix = format!("{key_name}:{minute}");
        let rpm_key = self.storage_key(rpm_suffix.as_bytes());
        let rpm_limit = self.requests_per_minute;

        let tpm_limit = self.tokens_per_minute;
        let tpm_key = tpm_limit.map(|_| {
            let tpm_suffix = format!("{key_name}:tpm:{minute}");
            self.storage_key(tpm_suffix.as_bytes())
        });

        Box::pin(async move {
            // Check RPM.
            let count = self
                .storage
                .increment(&rpm_key, 1)
                .await
                .map_err(|e| ExtensionError::new(500, e.to_string(), "server_error"))?;

            if count as u64 > rpm_limit {
                return Err(ExtensionError::new(
                    429,
                    "rate limit exceeded (RPM)",
                    "rate_limit_error",
                ));
            }

            // Check TPM.
            if let (Some(limit), Some(key)) = (tpm_limit, &tpm_key) {
                let tokens = self
                    .storage
                    .increment(key, 0)
                    .await
                    .map_err(|e| ExtensionError::new(500, e.to_string(), "server_error"))?;

                if tokens as u64 > limit {
                    return Err(ExtensionError::new(
                        429,
                        "rate limit exceeded (TPM)",
                        "rate_limit_error",
                    ));
                }
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
        if self.tokens_per_minute.is_none() {
            return Box::pin(async {});
        }

        let total_tokens = response
            .usage
            .as_ref()
            .map(|u| u.total_tokens as i64)
            .unwrap_or(0);

        if total_tokens == 0 {
            return Box::pin(async {});
        }

        let key_name = ctx.key_name.as_deref().unwrap_or("__global");
        let minute = current_minute();
        let tpm_suffix = format!("{key_name}:tpm:{minute}");
        let tpm_key = self.storage_key(tpm_suffix.as_bytes());

        Box::pin(async move {
            let _ = self.storage.increment(&tpm_key, total_tokens).await;
        })
    }

    fn on_chunk(&self, ctx: &RequestContext, chunk: &ChatCompletionChunk) -> BoxFuture<'_, ()> {
        if self.tokens_per_minute.is_none() {
            return Box::pin(async {});
        }

        let total_tokens = chunk
            .usage
            .as_ref()
            .map(|u| u.total_tokens as i64)
            .unwrap_or(0);

        if total_tokens == 0 {
            return Box::pin(async {});
        }

        let key_name = ctx.key_name.as_deref().unwrap_or("__global");
        let minute = current_minute();
        let tpm_suffix = format!("{key_name}:tpm:{minute}");
        let tpm_key = self.storage_key(tpm_suffix.as_bytes());

        Box::pin(async move {
            let _ = self.storage.increment(&tpm_key, total_tokens).await;
        })
    }
}

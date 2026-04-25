use crate::{PREFIX_KEYS, PREFIX_RATE_LIMIT};
use crabllm_core::{
    BoxFuture, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ExtensionError,
    KeyConfig, KeyRateLimit, RequestContext, Storage, storage_key,
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

    /// Look up per-key rate limit from storage. Returns the per-key
    /// override merged with global defaults, or the global defaults
    /// if the key has no override.
    async fn limits_for(&self, principal: &str) -> (u64, Option<u64>) {
        if principal == "__global" {
            return (self.requests_per_minute, self.tokens_per_minute);
        }

        let skey = storage_key(&PREFIX_KEYS, principal.as_bytes());
        let rl = self
            .storage
            .get(&skey)
            .await
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice::<KeyConfig>(&bytes).ok())
            .and_then(|kc| kc.rate_limit);

        match rl {
            Some(KeyRateLimit {
                requests_per_minute,
                tokens_per_minute,
            }) => (
                requests_per_minute.unwrap_or(self.requests_per_minute),
                tokens_per_minute.or(self.tokens_per_minute),
            ),
            None => (self.requests_per_minute, self.tokens_per_minute),
        }
    }
}

fn current_minute() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / 60
}

impl crabllm_core::Extension for RateLimit {
    fn name(&self) -> &str {
        "rate_limit"
    }

    fn prefix(&self) -> crabllm_core::Prefix {
        PREFIX_RATE_LIMIT
    }

    fn on_request(&self, ctx: &RequestContext) -> BoxFuture<'_, Result<(), ExtensionError>> {
        let principal = ctx.principal.as_deref().unwrap_or("__global").to_string();

        Box::pin(async move {
            let (rpm_limit, tpm_limit) = self.limits_for(&principal).await;
            let minute = current_minute();

            // Check RPM.
            let rpm_suffix = format!("{principal}:{minute}");
            let rpm_key = self.storage_key(rpm_suffix.as_bytes());
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
            if let Some(limit) = tpm_limit {
                let tpm_suffix = format!("{principal}:tpm:{minute}");
                let tpm_key = self.storage_key(tpm_suffix.as_bytes());
                let tokens = self
                    .storage
                    .increment(&tpm_key, 0)
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
        let total_tokens = response
            .usage
            .as_ref()
            .map(|u| u.total_tokens as i64)
            .unwrap_or(0);

        if total_tokens == 0 {
            return Box::pin(async {});
        }

        // Always record TPM — per-key overrides may enable TPM even when
        // the global config has no tokens_per_minute. The actual limit
        // check happens in on_request(); here we just track the counter.
        let principal = ctx.principal.as_deref().unwrap_or("__global");
        let minute = current_minute();
        let tpm_suffix = format!("{principal}:tpm:{minute}");
        let tpm_key = self.storage_key(tpm_suffix.as_bytes());

        Box::pin(async move {
            let _ = self.storage.increment(&tpm_key, total_tokens).await;
        })
    }

    fn on_chunk(&self, ctx: &RequestContext, chunk: &ChatCompletionChunk) -> BoxFuture<'_, ()> {
        let total_tokens = chunk
            .usage
            .as_ref()
            .map(|u| u.total_tokens as i64)
            .unwrap_or(0);

        if total_tokens == 0 {
            return Box::pin(async {});
        }

        let principal = ctx.principal.as_deref().unwrap_or("__global");
        let minute = current_minute();
        let tpm_suffix = format!("{principal}:tpm:{minute}");
        let tpm_key = self.storage_key(tpm_suffix.as_bytes());

        Box::pin(async move {
            let _ = self.storage.increment(&tpm_key, total_tokens).await;
        })
    }
}

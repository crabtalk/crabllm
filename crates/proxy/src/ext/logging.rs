use crabllm_core::{BoxFuture, Error, Prefix, RequestContext};

pub struct RequestLogger;

impl RequestLogger {
    pub fn new(_config: &serde_json::Value) -> Result<Self, String> {
        Ok(Self)
    }
}

impl crabllm_core::Extension for RequestLogger {
    fn name(&self) -> &str {
        "logging"
    }

    fn prefix(&self) -> Prefix {
        *b"logg"
    }

    fn on_response(
        &self,
        ctx: &RequestContext,
        _raw_request: &[u8],
        raw_response: &[u8],
    ) -> BoxFuture<'_, ()> {
        let latency = ctx.started_at.elapsed();
        let usage = crabllm_core::Usage::from(raw_response);

        tracing::info!(
            model = %ctx.model,
            provider = %ctx.provider,
            key = ctx.principal.as_deref().unwrap_or("-"),
            stream = ctx.is_stream,
            latency_ms = latency.as_millis() as u64,
            prompt_tokens = usage.prompt_tokens(),
            completion_tokens = usage.completion_tokens(),
            "request completed"
        );

        Box::pin(async {})
    }

    fn on_error(&self, ctx: &RequestContext, error: &Error) -> BoxFuture<'_, ()> {
        let latency = ctx.started_at.elapsed();

        tracing::warn!(
            model = %ctx.model,
            provider = %ctx.provider,
            key = ctx.principal.as_deref().unwrap_or("-"),
            stream = ctx.is_stream,
            latency_ms = latency.as_millis() as u64,
            error = %error,
            "request failed"
        );

        Box::pin(async {})
    }
}

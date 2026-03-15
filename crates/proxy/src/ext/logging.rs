use crabtalk_core::{BoxFuture, ChatCompletionResponse, Error, Prefix, RequestContext};

pub struct RequestLogger;

impl RequestLogger {
    pub fn new(_config: &toml::Value) -> Result<Self, String> {
        Ok(Self)
    }
}

impl crabtalk_core::Extension for RequestLogger {
    fn name(&self) -> &str {
        "logging"
    }

    fn prefix(&self) -> Prefix {
        *b"logg"
    }

    fn on_response(
        &self,
        ctx: &RequestContext,
        response: &ChatCompletionResponse,
    ) -> BoxFuture<'_, ()> {
        let latency = ctx.started_at.elapsed();
        let (prompt, completion) = response
            .usage
            .as_ref()
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        tracing::info!(
            model = %ctx.model,
            provider = %ctx.provider,
            key = ctx.key_name.as_deref().unwrap_or("-"),
            stream = ctx.is_stream,
            latency_ms = latency.as_millis() as u64,
            prompt_tokens = prompt,
            completion_tokens = completion,
            "request completed"
        );

        Box::pin(async {})
    }

    fn on_error(&self, ctx: &RequestContext, error: &Error) -> BoxFuture<'_, ()> {
        let latency = ctx.started_at.elapsed();

        tracing::warn!(
            model = %ctx.model,
            provider = %ctx.provider,
            key = ctx.key_name.as_deref().unwrap_or("-"),
            stream = ctx.is_stream,
            latency_ms = latency.as_millis() as u64,
            error = %error,
            "request failed"
        );

        Box::pin(async {})
    }
}

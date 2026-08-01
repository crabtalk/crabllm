use crate::{ApiError, BoxFuture, Error, Prefix, storage_key};
use std::time::Instant;

/// Per-request metadata passed to extension hooks.
#[derive(Clone, Debug)]
pub struct RequestContext {
    pub request_id: String,
    pub model: String,
    pub provider: String,
    /// Opaque identity token attached by the authentication layer. Treat as
    /// opaque — do not parse, sanitize, or display without intentional formatting.
    pub principal: Option<String>,
    pub is_stream: bool,
    pub started_at: Instant,
}

/// Error returned by `Extension::on_request` to short-circuit the pipeline.
/// Converted to an HTTP response in the handler.
///
/// `body` is the gateway's own envelope, which suits an extension whose
/// refusals are the gateway's to describe. An extension written for one
/// deployment usually isn't: its clients already switch on that product's
/// error contract, and answering in a second shape for these few codes would
/// make them handle both. Such an extension sets [`json`](Self::json) and
/// [`headers`](Self::headers) to answer in the shape its callers expect.
pub struct ExtensionError {
    pub status: u16,
    pub body: ApiError,
    /// Serialized in place of `body` when set.
    pub json: Option<serde_json::Value>,
    /// Extra response headers, e.g. `retry-after` on a quota refusal.
    pub headers: Vec<(String, String)>,
}

impl ExtensionError {
    pub fn new(status: u16, message: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            status,
            body: ApiError::new(message, kind),
            json: None,
            headers: Vec::new(),
        }
    }

    /// Answer with `json` verbatim instead of the gateway's envelope.
    pub fn with_json(mut self, json: serde_json::Value) -> Self {
        self.json = Some(json);
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The body to serialize: the override when present, the envelope
    /// otherwise. Callers render the response; core stays free of a web
    /// framework.
    pub fn into_json(self) -> (u16, serde_json::Value, Vec<(String, String)>) {
        let body = self
            .json
            .unwrap_or_else(|| serde_json::to_value(&self.body).unwrap_or(serde_json::Value::Null));
        (self.status, body, self.headers)
    }
}

/// Trait for request pipeline extensions (usage tracking, logging, rate limiting, etc.).
///
/// Extensions receive raw bytes — they deserialize only the fields they need.
/// All methods have default no-op implementations except `name` and `prefix`.
///
/// Extensions must be `Send + Sync` for use across async handler tasks.
/// Hook methods return `BoxFuture` for dyn-compatibility.
pub trait Extension: Send + Sync {
    /// Human-readable name for this extension, used in logs and diagnostics.
    fn name(&self) -> &str;

    /// Fixed 4-byte prefix that namespaces this extension's storage keys.
    fn prefix(&self) -> Prefix;

    /// Build a full storage key by prepending this extension's prefix to `suffix`.
    fn storage_key(&self, suffix: &[u8]) -> Vec<u8> {
        storage_key(&self.prefix(), suffix)
    }

    /// Check for a cached response before provider dispatch. Return `Some`
    /// with raw response bytes to skip the provider call entirely.
    /// Called for non-streaming requests only.
    fn on_cache_lookup(&self, _raw_request: &[u8]) -> BoxFuture<'_, Option<Vec<u8>>> {
        Box::pin(async { None })
    }

    /// Called post-auth, pre-dispatch. Return `Err` to short-circuit the pipeline
    /// (no provider call, no further extensions run).
    fn on_request(&self, _ctx: &RequestContext) -> BoxFuture<'_, Result<(), ExtensionError>> {
        Box::pin(async { Ok(()) })
    }

    /// Called after a non-streaming response arrives from the provider.
    /// Both request and response are raw wire bytes.
    fn on_response(
        &self,
        _ctx: &RequestContext,
        _raw_request: &[u8],
        _raw_response: &[u8],
    ) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    /// Called once per SSE chunk during a streaming response.
    /// `raw_chunk` is the serialized JSON of the chunk.
    fn on_chunk(&self, _ctx: &RequestContext, _raw_chunk: &[u8]) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    /// Called when the provider returns an error.
    fn on_error(&self, _ctx: &RequestContext, _error: &Error) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

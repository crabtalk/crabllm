use crate::http::Http;

/// Anthropic API version sent with `/v1/messages` requests, matching the value
/// the native Anthropic API and crabllm's own Anthropic provider use.
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";

/// A typed client for a crabllm-compatible gateway (e.g. a deployed
/// `crabllm-proxy`).
///
/// `Client` implements [`crabllm_core::Provider`], so it composes with
/// [`crabllm_core::Retrying`] for retries and is a drop-in anywhere a
/// `Provider` is expected. Cloning is cheap — the inner HTTP client is
/// `Arc`-shared by reqwest.
#[derive(Clone, Debug)]
pub struct Client {
    base_url: String,
    api_key: String,
    pub(crate) http: Http,
}

impl Client {
    /// Build a client for `base_url` (the gateway origin, e.g.
    /// `https://api.example.com`) authenticating with `api_key`. Any trailing
    /// slash on `base_url` is trimmed; endpoint paths like `/v1/messages` are
    /// appended per request.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            http: Http::new(),
        }
    }

    /// Full URL for a gateway path (path starts with `/`).
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// `Authorization` header value. The gateway accepts `Bearer <key>`.
    pub(crate) fn bearer(&self) -> String {
        format!("Bearer {}", self.api_key)
    }
}

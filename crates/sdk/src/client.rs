use bytes::Bytes;
use crabllm_core::{ByteStream, Error, Retrying};
use crabllm_http::HttpClient;
use std::time::Duration;

/// `content-type` sent on every request.
pub(crate) const JSON: &str = "application/json";

/// Anthropic API version sent with `/v1/messages` requests, matching the value
/// the native Anthropic API and crabllm's own Anthropic provider use.
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";

/// How the client authenticates to the gateway. The gateway accepts both, so
/// pick whichever matches the calling convention you're emulating.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Auth {
    /// `Authorization: Bearer <key>` — the OpenAI convention.
    #[default]
    Bearer,
    /// `x-api-key: <key>` — the Anthropic convention.
    ApiKey,
}

/// The bare gateway client: serializes typed requests, POSTs them, and parses
/// the response. Implements [`crabllm_core::Provider`] with no retry of its own
/// — [`Client`] layers retry on top via [`Retrying`].
#[derive(Debug, Clone)]
pub(crate) struct RawClient {
    base_url: String,
    api_key: String,
    auth: Auth,
    pub(crate) http: HttpClient,
}

impl RawClient {
    pub(crate) fn new(base_url: String, api_key: String, auth: Auth) -> Self {
        // `base_url` is the gateway *origin*; the SDK owns the route paths
        // (`/v1/...` and `/v1beta/...`), so a single `/v1`-inclusive base
        // can't serve the gemini route. A trailing `/v1` is tolerated and
        // stripped so existing OpenAI/Anthropic-style configs don't become
        // `…/v1/v1/messages`.
        let trimmed = base_url.trim_end_matches('/');
        let origin = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
        Self {
            base_url: origin.to_string(),
            api_key,
            auth,
            http: HttpClient::new(),
        }
    }

    /// Full URL for a gateway path (path starts with `/`).
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// The auth header (name, value) for the configured scheme.
    fn auth_header(&self) -> (&'static str, String) {
        match self.auth {
            Auth::Bearer => ("authorization", format!("Bearer {}", self.api_key)),
            Auth::ApiKey => ("x-api-key", self.api_key.clone()),
        }
    }

    /// POST a JSON body and return the response bytes, mapping a gateway
    /// 4xx/5xx to [`Error::Provider`] with its body verbatim. `extra` carries
    /// per-endpoint headers (e.g. `anthropic-version`).
    pub(crate) async fn post_checked(
        &self,
        path: &str,
        extra: &[(&str, &str)],
        body: Bytes,
    ) -> Result<Bytes, Error> {
        let (name, value) = self.auth_header();
        let mut headers = vec![("content-type", JSON), (name, value.as_str())];
        headers.extend_from_slice(extra);
        let resp = self.http.post(&self.url(path), &headers, body).await?;
        if resp.status >= 400 {
            return Err(Error::Provider {
                status: resp.status,
                body: String::from_utf8_lossy(&resp.body).into_owned(),
                retry_after: resp.retry_after,
            });
        }
        Ok(resp.body)
    }

    /// POST a JSON body and return the raw SSE byte stream. A 4xx/5xx is
    /// already mapped to [`Error::Provider`] by the transport.
    pub(crate) async fn post_sse(
        &self,
        path: &str,
        extra: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        let (name, value) = self.auth_header();
        let mut headers = vec![("content-type", JSON), (name, value.as_str())];
        headers.extend_from_slice(extra);
        self.http.post_stream(&self.url(path), &headers, body).await
    }
}

/// A typed client for a crabllm-compatible gateway (a deployed `crabllm-proxy`).
///
/// Implements [`crabllm_core::Provider`], so it drops into any `Provider`-generic
/// code. Retries (transient failures, exponential backoff + jitter) and a
/// per-attempt timeout are **on by default** — configure or disable them via
/// [`Client::builder`]. Streaming responses are parsed by the shared
/// `crabllm_core::codec`, the same code the gateway uses. Cloning is cheap.
#[derive(Debug, Clone)]
pub struct Client {
    pub(crate) inner: Retrying<RawClient>,
}

impl Client {
    /// Client for `base_url` with `api_key`, `Bearer` auth, and the default
    /// retry policy (2 retries, 30s per-attempt timeout).
    ///
    /// `base_url` is the gateway **origin** (e.g. `https://api.example.com`) —
    /// the SDK appends the route paths (`/v1/messages`, `/v1beta/...`). A
    /// trailing `/v1` is accepted and stripped.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::builder(base_url, api_key).build()
    }

    /// Start configuring a client — auth scheme, retries, timeout.
    pub fn builder(base_url: impl Into<String>, api_key: impl Into<String>) -> ClientBuilder {
        ClientBuilder {
            base_url: base_url.into(),
            api_key: api_key.into(),
            auth: Auth::default(),
            max_retries: None,
            timeout: None,
            max_retry_after: None,
        }
    }
}

/// Builder for [`Client`]. Unset knobs keep [`Retrying`]'s defaults.
pub struct ClientBuilder {
    base_url: String,
    api_key: String,
    auth: Auth,
    max_retries: Option<u32>,
    timeout: Option<Duration>,
    max_retry_after: Option<Duration>,
}

impl ClientBuilder {
    /// Authentication scheme (default [`Auth::Bearer`]).
    pub fn auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Maximum retries for transient failures. `0` disables retrying.
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Per-attempt timeout. Zero disables it.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// Largest `Retry-After` the client will honor before giving up.
    pub fn max_retry_after(mut self, d: Duration) -> Self {
        self.max_retry_after = Some(d);
        self
    }

    pub fn build(self) -> Client {
        let raw = RawClient::new(self.base_url, self.api_key, self.auth);
        let mut retrying = Retrying::new(raw);
        if let Some(n) = self.max_retries {
            retrying = retrying.max_retries(n);
        }
        if let Some(t) = self.timeout {
            retrying = retrying.timeout(t);
        }
        if let Some(m) = self.max_retry_after {
            retrying = retrying.max_retry_after(m);
        }
        Client { inner: retrying }
    }
}

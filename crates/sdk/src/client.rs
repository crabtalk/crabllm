use bytes::Bytes;
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ByteStream,
    ChatCompletionRequest, Dialect, Error, ModelList, Provider, Retrying,
    codec::anthropic::chunks_to_anthropic_events, ir,
};
use crabllm_http::HttpClient;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// `content-type` sent on every request.
pub(crate) const JSON: &str = "application/json";

/// Shared `model → native dialects` cache, populated lazily from `/v1/models`.
type DialectCache = Arc<RwLock<Option<HashMap<String, Vec<Dialect>>>>>;

/// How `bridge` mode dispatches an Anthropic request, given the target model's
/// native dialects. Exposed (with [`route`]) so callers can predict or test the
/// routing without issuing a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Forward 1:1 to `/v1/messages` — the model is Anthropic-native.
    Native,
    /// Translate through `/v1/chat/completions` — no native Anthropic endpoint.
    Translate,
    /// Unknown model (catalog miss/fetch failure): try native, else translate.
    NativeElseTranslate,
}

/// Decide how to route an Anthropic request for a model with these native
/// dialects. Anthropic-native wins (full fidelity); an OpenAI-only model
/// translates; anything else is unknown and tries native first.
pub fn route(dialects: &[Dialect]) -> Route {
    if dialects.contains(&Dialect::Anthropic) {
        Route::Native
    } else if dialects.contains(&Dialect::Openai) {
        Route::Translate
    } else {
        Route::NativeElseTranslate
    }
}

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

    /// `GET /v1/models`. Returns the OpenAI-shaped [`ModelList`]; with
    /// [`Auth::ApiKey`] the gateway answers in Anthropic shape instead, so this
    /// is meaningful only for the default [`Auth::Bearer`].
    pub(crate) async fn models(&self) -> Result<ModelList, Error> {
        let (name, value) = self.auth_header();
        let headers = [("content-type", JSON), (name, value.as_str())];
        let resp = self.http.get(&self.url("/v1/models"), &headers).await?;
        if resp.status >= 400 {
            return Err(Error::Provider {
                status: resp.status,
                body: String::from_utf8_lossy(&resp.body).into_owned(),
                retry_after: resp.retry_after,
            });
        }
        crabllm_core::json::from_slice(&resp.body).map_err(|e| Error::Decode(e.to_string()))
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
    /// When set, `anthropic_messages`/`_stream` route by the model's native
    /// dialects (`/v1/models`): forward 1:1 to `/v1/messages` if the model is
    /// Anthropic-native, otherwise translate through `/v1/chat/completions`.
    /// Off by default — the client forwards 1:1 with no translation.
    pub(crate) bridge: bool,
    /// Cached `model → native dialects`, populated from `/v1/models` on first
    /// need. `None` until fetched; shared across clones.
    pub(crate) dialects: DialectCache,
}

impl Client {
    /// Native dialects for `model`, from `/v1/models` (cached). Empty if the
    /// model isn't listed or the catalog couldn't be fetched — the caller then
    /// treats it as unknown (try native, fall back to translation).
    ///
    /// The catalog is fetched once and never invalidated for the client's
    /// lifetime. That's fine for models added after the fetch (they read as
    /// unknown and take the fallback path), but a model that was cached as
    /// Anthropic-native and later loses that endpoint would forward natively
    /// and fail with no fallback. Recreate the client if the deployment's model
    /// routing changes under it. Bridging assumes `Auth::Bearer`; with
    /// `Auth::ApiKey` the gateway answers `/v1/models` in Anthropic shape, the
    /// parse below fails, and every model degrades to the unknown fallback.
    pub(crate) async fn model_dialects(&self, model: &str) -> Vec<Dialect> {
        // Read under an explicit scope so the guard can't be held across the
        // `.await` below, no matter how this is edited later.
        {
            let cache = self.dialects.read().unwrap();
            if let Some(map) = cache.as_ref() {
                return map.get(model).cloned().unwrap_or_default();
            }
        }
        // First need: fetch the catalog once. On failure, cache an empty map so
        // we don't refetch on every call, and warn rather than degrade in
        // silence — every model then takes the unknown (native-then-translate)
        // path.
        let map: HashMap<String, Vec<Dialect>> = match self.models().await {
            Ok(list) => list.data.into_iter().map(|m| (m.id, m.dialects)).collect(),
            Err(e) => {
                tracing::warn!(
                    "dialect capability fetch from /v1/models failed; bridging degraded: {e}"
                );
                HashMap::new()
            }
        };
        let hit = map.get(model).cloned().unwrap_or_default();
        *self.dialects.write().unwrap() = Some(map);
        hit
    }

    /// Translate an Anthropic request through the OpenAI endpoint:
    /// `AnthropicRequest → IR → chat completion → IR → AnthropicResponse`.
    /// Lossy (the IR normalizes), so it's the fallback for OpenAI-only models,
    /// never the path for Anthropic-native ones.
    pub(crate) async fn translate_anthropic(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let chat_req = ChatCompletionRequest::from(&ir::Request::from(request.clone()));
        let chat_resp = self.inner.chat_completion(&chat_req).await?;
        Ok(AnthropicResponse::from(&ir::Response::from(chat_resp)))
    }

    /// Streaming counterpart: chat-completion chunks re-encoded as Anthropic
    /// stream events.
    pub(crate) async fn translate_anthropic_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let chat_req = ChatCompletionRequest::from(&ir::Request::from(request.clone()));
        let chunks = self.inner.chat_completion_stream(&chat_req).await?;
        Ok(chunks_to_anthropic_events(chunks).boxed())
    }
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

    /// List the models the gateway exposes (`GET /v1/models`). Not retried —
    /// it's an idempotent listing, and not part of the `Provider` trait.
    pub async fn models(&self) -> Result<ModelList, Error> {
        self.inner.get_ref().models().await
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
            bridge: false,
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
    bridge: bool,
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

    /// Enable dialect bridging: `anthropic_messages`/`_stream` consult
    /// `/v1/models` and translate through `/v1/chat/completions` for models
    /// that aren't Anthropic-native. Off by default (1:1 forwarding). Native
    /// models always pass through untranslated; translation is lossy, so it's
    /// only used when a model has no native Anthropic endpoint.
    pub fn bridge(mut self, on: bool) -> Self {
        self.bridge = on;
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
        Client {
            inner: retrying,
            bridge: self.bridge,
            dialects: Arc::new(RwLock::new(None)),
        }
    }
}

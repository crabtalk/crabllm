use arc_swap::ArcSwap;
use crabllm_core::{Extension, GatewayConfig, Provider, Storage};
use crabllm_provider::ProviderRegistry;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::SystemTime,
};
use tokio::sync::broadcast;

/// Per-request event emitted after a request completes. Embedders
/// subscribe to the [`AppState::usage_events`] broadcast channel to
/// observe live traffic without scraping the Prometheus endpoint.
#[derive(Debug, Clone)]
pub struct UsageEvent {
    pub timestamp: SystemTime,
    pub request_id: String,
    pub principal: Option<String>,
    pub model: String,
    pub provider: String,
    /// Logical endpoint: `"chat.completions"`, `"embeddings"`,
    /// `"images.generations"`, `"audio.speech"`, `"audio.transcriptions"`.
    pub endpoint: &'static str,
    /// Prompt / input tokens. For embeddings this is the input token
    /// count; for image / audio endpoints it's 0.
    pub tokens_in: u32,
    /// Completion / output tokens. 0 for endpoints that don't generate
    /// tokens (embeddings, images, audio).
    pub tokens_out: u32,
    pub duration_ms: u64,
    /// The wire HTTP status the client observed, or `0` when a
    /// streaming chat response sent 200 OK headers and then broke
    /// mid-stream. `0` is a sentinel meaning "not a real HTTP
    /// response" — consumers branching on `status` alone can
    /// distinguish a clean 200 from a failed stream without having
    /// to inspect [`Self::error`]. For non-streaming requests
    /// `status` is always the real HTTP code the client saw.
    pub status: u16,
    /// `Some(msg)` if the request failed. Set alongside `status == 0`
    /// for mid-stream streaming failures; set alongside a real error
    /// status (4xx/5xx) for pre-stream and non-streaming failures.
    pub error: Option<String>,
}

/// Shared application state passed to all handlers.
///
/// Generic over the storage backend `S` and the provider type `P`. The
/// binary picks `P` by defining a workspace-level union enum that wraps
/// every provider source it links — that enum implements `Provider` via
/// match-and-delegate, so dispatch through `P` is fully monomorphized.
pub struct AppState<S: Storage, P: Provider> {
    pub registry: Arc<ArcSwap<ProviderRegistry<P>>>,
    pub config: GatewayConfig,
    pub extensions: Arc<Vec<Box<dyn Extension>>>,
    pub storage: Arc<S>,
    /// Precomputed token → key name lookup for O(1) auth.
    /// Wrapped in RwLock to support runtime key management.
    pub key_map: Arc<RwLock<HashMap<String, String>>>,
    /// Optional broadcast sink for per-request [`UsageEvent`]s. `None`
    /// is a no-op — the standalone `crabllm serve` binary leaves it
    /// unset and behavior is unchanged. Embedders that want live
    /// traffic construct a sender, pass `Some(sender.clone())`, and
    /// call `sender.subscribe()` to observe events.
    pub usage_events: Option<broadcast::Sender<UsageEvent>>,
}

impl<S: Storage, P: Provider> Clone for AppState<S, P> {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            config: self.config.clone(),
            extensions: self.extensions.clone(),
            storage: self.storage.clone(),
            key_map: self.key_map.clone(),
            usage_events: self.usage_events.clone(),
        }
    }
}

impl<S: Storage, P: Provider> AppState<S, P> {
    /// Snapshot the current provider registry. Returns a guard that
    /// keeps the referenced registry alive for the duration of the
    /// request — even if a concurrent swap replaces the global pointer.
    pub fn registry(&self) -> arc_swap::Guard<Arc<ProviderRegistry<P>>> {
        self.registry.load()
    }
}

use crabllm_core::{Extension, GatewayConfig, Provider, Storage};
use crabllm_provider::ProviderRegistry;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// Shared application state passed to all handlers.
///
/// Generic over the storage backend `S` and the provider type `P`. The
/// binary picks `P` by defining a workspace-level union enum that wraps
/// every provider source it links — that enum implements `Provider` via
/// match-and-delegate, so dispatch through `P` is fully monomorphized.
pub struct AppState<S: Storage, P: Provider> {
    pub registry: ProviderRegistry<P>,
    pub config: GatewayConfig,
    pub extensions: Arc<Vec<Box<dyn Extension>>>,
    pub storage: Arc<S>,
    /// Precomputed token → key name lookup for O(1) auth.
    /// Wrapped in RwLock to support runtime key management.
    pub key_map: Arc<RwLock<HashMap<String, String>>>,
}

impl<S: Storage, P: Provider> Clone for AppState<S, P> {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            config: self.config.clone(),
            extensions: self.extensions.clone(),
            storage: self.storage.clone(),
            key_map: self.key_map.clone(),
        }
    }
}

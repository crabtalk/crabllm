use crabtalk_core::{Extension, GatewayConfig, Storage};
use crabtalk_provider::ProviderRegistry;
use std::{collections::HashMap, sync::Arc};

/// Shared application state passed to all handlers.
pub struct AppState<S: Storage> {
    pub registry: ProviderRegistry,
    pub client: reqwest::Client,
    pub config: GatewayConfig,
    pub extensions: Arc<Vec<Box<dyn Extension>>>,
    pub storage: Arc<S>,
    /// Precomputed token → key name lookup for O(1) auth.
    pub key_map: HashMap<String, String>,
}

impl<S: Storage> Clone for AppState<S> {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            client: self.client.clone(),
            config: self.config.clone(),
            extensions: self.extensions.clone(),
            storage: self.storage.clone(),
            key_map: self.key_map.clone(),
        }
    }
}

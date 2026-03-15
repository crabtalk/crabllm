use std::sync::Arc;

use crabtalk_core::{Extension, GatewayConfig, Storage};
use crabtalk_provider::ProviderRegistry;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: ProviderRegistry,
    pub client: reqwest::Client,
    pub config: GatewayConfig,
    pub extensions: Arc<Vec<Box<dyn Extension>>>,
    pub storage: Arc<dyn Storage>,
}

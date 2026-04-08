use crate::{RemoteProvider, make_client};
use bytes::Bytes;
use crabllm_core::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, GatewayConfig,
    ImageRequest, MultipartField, Provider, ProviderConfig, ProviderKind,
};
use rand::Rng;
use std::{collections::HashMap, sync::Arc, time::Duration};

/// A provider entry with its routing weight and retry config.
///
/// Generic over `P` so the binary can supply any concrete provider type
/// implementing `crabllm_core::Provider` — typically a workspace-level
/// union enum that delegates to multiple sources (built-in remote APIs,
/// local backends, etc.).
///
/// Note: `Deployment` does not derive `Clone`. The registry stores
/// `Arc<Deployment<P>>` internally so multiple model → deployment bindings
/// share a single allocation, and so `P` is free to hold non-`Clone` state
/// (e.g. model mmaps, CUDA contexts, task handles).
#[derive(Debug)]
pub struct Deployment<P> {
    pub provider: P,
    pub weight: u16,
    pub max_retries: u32,
    pub timeout: Duration,
}

/// Maps model names to weighted provider lists for routing.
#[derive(Debug)]
pub struct ProviderRegistry<P> {
    providers: HashMap<String, Vec<Arc<Deployment<P>>>>,
    aliases: HashMap<String, String>,
    /// Precomputed model name → provider name lookup (avoids per-request HashMap rebuild).
    model_providers: HashMap<String, String>,
}

/// Manual `Clone` impl so the bound is not `P: Clone` — only the outer
/// `HashMap`/`Vec`/`Arc` layers clone, and `Arc::clone` is infallible for
/// any `T`. Cloning the registry is a shallow structural copy; every
/// clone shares the same underlying `Deployment` allocations.
impl<P> Clone for ProviderRegistry<P> {
    fn clone(&self) -> Self {
        Self {
            providers: self.providers.clone(),
            aliases: self.aliases.clone(),
            model_providers: self.model_providers.clone(),
        }
    }
}

impl<P> ProviderRegistry<P> {
    /// Create a registry directly from pre-built provider lists and aliases.
    pub fn new(
        providers: HashMap<String, Vec<Arc<Deployment<P>>>>,
        aliases: HashMap<String, String>,
        model_providers: HashMap<String, String>,
    ) -> Self {
        ProviderRegistry {
            providers,
            aliases,
            model_providers,
        }
    }

    /// Insert a pre-built deployment for a single model. Maintains the
    /// invariant that every key in `model_providers` has a matching entry
    /// in `providers`.
    ///
    /// Used by the binary to attach a locally-built provider (e.g. the
    /// llama.cpp backend) to a registry that was first populated from
    /// remote-provider configs via [`from_provider_configs`](Self::from_provider_configs).
    pub fn insert_deployment(
        &mut self,
        model: String,
        provider_name: String,
        deployment: Deployment<P>,
    ) {
        let arc = Arc::new(deployment);
        self.providers.entry(model.clone()).or_default().push(arc);
        self.model_providers.insert(model, provider_name);
    }

    /// Look up the provider name for a model. O(1) HashMap lookup.
    pub fn provider_name(&self, model: &str) -> Option<&str> {
        self.model_providers.get(model).map(|s| s.as_str())
    }

    /// Return all registered model names.
    pub fn model_names(&self) -> impl Iterator<Item = &str> {
        self.model_providers.keys().map(|s| s.as_str())
    }

    /// Count distinct registered provider names (remote + local combined).
    pub fn provider_count(&self) -> usize {
        self.model_providers
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Resolve a model name through aliases. Returns the canonical name.
    pub fn resolve<'a>(&'a self, model: &'a str) -> &'a str {
        self.aliases.get(model).map(|s| s.as_str()).unwrap_or(model)
    }

    /// Select a provider for a model using weighted random selection.
    /// Returns None if the model is not registered.
    pub fn dispatch(&self, model: &str) -> Option<&Deployment<P>> {
        let list = self.providers.get(model)?;
        if list.len() == 1 {
            return Some(&list[0]);
        }

        let total: u32 = list.iter().map(|d| d.weight as u32).sum();
        if total == 0 {
            return Some(&list[0]);
        }

        let mut pick = rand::rng().random_range(0..total);
        for d in list {
            let w = d.weight as u32;
            if pick < w {
                return Some(d);
            }
            pick -= w;
        }

        // Fallback (shouldn't happen).
        Some(&list[0])
    }

    /// Return all deployments for a model, ordered for fallback:
    /// selected provider first, then remaining sorted by descending weight.
    /// Returns None if the model is not registered.
    pub fn dispatch_list(&self, model: &str) -> Option<Vec<&Deployment<P>>> {
        let list = self.providers.get(model)?;
        if list.len() == 1 {
            return Some(vec![&list[0]]);
        }

        let total: u32 = list.iter().map(|d| d.weight as u32).sum();
        let selected_idx = if total == 0 {
            0
        } else {
            let mut pick = rand::rng().random_range(0..total);
            let mut idx = 0;
            for (i, d) in list.iter().enumerate() {
                let w = d.weight as u32;
                if pick < w {
                    idx = i;
                    break;
                }
                pick -= w;
            }
            idx
        };

        // Build result: selected first, then remaining sorted by descending weight.
        let mut result: Vec<&Deployment<P>> = Vec::with_capacity(list.len());
        result.push(&list[selected_idx]);

        let mut remaining: Vec<(usize, &Arc<Deployment<P>>)> = list
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != selected_idx)
            .collect();
        remaining.sort_by(|a, b| b.1.weight.cmp(&a.1.weight));
        result.extend(remaining.into_iter().map(|(_, d)| d.as_ref()));

        Some(result)
    }

    /// Check if a model is registered (after alias resolution).
    pub fn has_model(&self, model: &str) -> bool {
        self.providers.contains_key(model)
    }
}

impl<P> ProviderRegistry<P> {
    /// Build the registry from a full `GatewayConfig`. Thin wrapper around
    /// [`from_provider_configs`](Self::from_provider_configs) that pulls the
    /// two fields the registry actually cares about.
    pub fn from_config<F>(config: &GatewayConfig, wrap: F) -> Result<Self, Error>
    where
        F: Fn(RemoteProvider) -> P,
    {
        Self::from_provider_configs(&config.providers, &config.aliases, wrap)
    }

    /// Build the registry directly from a provider map and an alias map,
    /// without requiring a full `GatewayConfig`. Intended for library
    /// consumers that construct a handful of providers programmatically and
    /// don't carry the rest of the gateway's runtime config
    /// (listen address, storage, extensions, …).
    ///
    /// `wrap` lifts each constructed `RemoteProvider` into the workspace-
    /// level concrete type the caller uses as `P` — e.g., a union enum
    /// with one variant per provider source, or the identity closure when
    /// `P = RemoteProvider`.
    pub fn from_provider_configs<F>(
        providers_config: &HashMap<String, ProviderConfig>,
        aliases: &HashMap<String, String>,
        wrap: F,
    ) -> Result<Self, Error>
    where
        F: Fn(RemoteProvider) -> P,
    {
        // Validate against the raw config so we don't waste a `reqwest::Client`
        // construction (TLS init) on inputs we'd reject.
        for (provider_name, provider_config) in providers_config {
            validate_provider(provider_name, provider_config)?;
        }

        // One shared `reqwest::Client` — cheap to clone, so every provider
        // dispatches through the same connection pool.
        let client = make_client();

        let mut providers: HashMap<String, Vec<Arc<Deployment<P>>>> = HashMap::new();

        for provider_config in providers_config.values() {
            let provider = wrap(RemoteProvider::new(provider_config, client.clone()));

            let deployment = Arc::new(Deployment {
                provider,
                weight: provider_config.weight.unwrap_or(1),
                max_retries: provider_config.max_retries.unwrap_or(2),
                timeout: Duration::from_secs(provider_config.timeout.unwrap_or(30)),
            });
            for model_name in &provider_config.models {
                providers
                    .entry(model_name.clone())
                    .or_default()
                    .push(Arc::clone(&deployment));
            }
        }

        let mut model_providers = HashMap::new();
        for (provider_name, provider_config) in providers_config {
            for model in &provider_config.models {
                model_providers.insert(model.clone(), provider_name.clone());
            }
        }

        Ok(Self::new(providers, aliases.clone(), model_providers))
    }
}

/// Validate provider-specific required fields against the raw config.
fn validate_provider(name: &str, config: &ProviderConfig) -> Result<(), Error> {
    fn is_blank(opt: &Option<String>) -> bool {
        opt.as_ref().is_none_or(|s| s.is_empty())
    }
    match config.kind {
        ProviderKind::Openai | ProviderKind::Ollama => {
            // Both have a sensible default base_url; nothing to require.
            Ok(())
        }
        ProviderKind::Anthropic | ProviderKind::Google => {
            if is_blank(&config.api_key) {
                return Err(Error::Config(format!(
                    "provider '{name}' ({:?}) requires an api_key",
                    config.kind,
                )));
            }
            Ok(())
        }
        ProviderKind::Azure => {
            if is_blank(&config.api_key) {
                return Err(Error::Config(format!(
                    "provider '{name}' (azure) requires an api_key"
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "provider-bedrock"))]
        ProviderKind::Bedrock => Err(Error::Config(format!(
            "provider '{name}' uses kind = 'bedrock', which requires the \
             'provider-bedrock' feature to be enabled in the crabllm binary"
        ))),
        #[cfg(feature = "provider-bedrock")]
        ProviderKind::Bedrock => {
            if is_blank(&config.region) {
                return Err(Error::Config(format!(
                    "provider '{name}' (bedrock) requires a region"
                )));
            }
            if is_blank(&config.access_key) {
                return Err(Error::Config(format!(
                    "provider '{name}' (bedrock) requires an access_key"
                )));
            }
            if is_blank(&config.secret_key) {
                return Err(Error::Config(format!(
                    "provider '{name}' (bedrock) requires a secret_key"
                )));
            }
            Ok(())
        }
    }
}

/// `ProviderRegistry<P>` is itself a `Provider`. Each call routes on the
/// request's (alias-resolved) model name, picks one weighted deployment via
/// `dispatch`, and forwards to the inner provider.
///
/// This lets downstream library consumers use the registry directly as a
/// `P: Provider` — the registry handles model-name routing so the caller
/// never has to touch `Deployment` or write their own delegation wrapper.
///
/// Note: this impl picks a single deployment and does not walk fallback
/// deployments or retry. The HTTP proxy still drives its own retry/fallback
/// loops over `&Deployment<P>` from `dispatch_list`, and does not dispatch
/// through this impl.
impl<P: Provider> Provider for ProviderRegistry<P> {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let model = self.resolve(&request.model);
        let deployment = self
            .dispatch(model)
            .ok_or_else(|| model_not_registered(model))?;
        deployment.provider.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let model = self.resolve(&request.model);
        let deployment = self
            .dispatch(model)
            .ok_or_else(|| model_not_registered(model))?;
        deployment.provider.chat_completion_stream(request).await
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        let model = self.resolve(&request.model);
        let deployment = self
            .dispatch(model)
            .ok_or_else(|| model_not_registered(model))?;
        deployment.provider.embedding(request).await
    }

    async fn image_generation(&self, request: &ImageRequest) -> Result<(Bytes, String), Error> {
        let model = self.resolve(&request.model);
        let deployment = self
            .dispatch(model)
            .ok_or_else(|| model_not_registered(model))?;
        deployment.provider.image_generation(request).await
    }

    async fn audio_speech(&self, request: &AudioSpeechRequest) -> Result<(Bytes, String), Error> {
        let model = self.resolve(&request.model);
        let deployment = self
            .dispatch(model)
            .ok_or_else(|| model_not_registered(model))?;
        deployment.provider.audio_speech(request).await
    }

    async fn audio_transcription(
        &self,
        model: &str,
        fields: &[MultipartField],
    ) -> Result<(Bytes, String), Error> {
        let resolved = self.resolve(model);
        let deployment = self
            .dispatch(resolved)
            .ok_or_else(|| model_not_registered(resolved))?;
        deployment.provider.audio_transcription(model, fields).await
    }
}

fn model_not_registered(model: &str) -> Error {
    // Not `Error::Config`: this is a runtime routing miss, not a TOML parse
    // problem. The taxonomy is missing a real `NotFound` variant — see the
    // follow-up issue. `Internal` is the lesser evil among existing variants.
    Error::Internal(format!(
        "model '{model}' not registered in provider registry"
    ))
}

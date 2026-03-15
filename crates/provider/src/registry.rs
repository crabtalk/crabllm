use crate::Provider;
use crabtalk_core::{Error, GatewayConfig, ProviderKind};
use std::collections::HashMap;

/// Maps model names to configured Provider instances.
#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    providers: HashMap<String, Provider>,
}

impl ProviderRegistry {
    /// Build the registry from gateway config.
    /// Returns an error if a model route references an unknown provider
    /// or if a provider is missing a required api_key.
    pub fn from_config(config: &GatewayConfig) -> Result<Self, Error> {
        let mut providers = HashMap::new();

        for (model_name, route) in &config.models {
            let provider_config = config.providers.get(&route.provider).ok_or_else(|| {
                Error::Config(format!(
                    "model '{}' references unknown provider '{}'",
                    model_name, route.provider
                ))
            })?;

            let provider = match provider_config.kind {
                ProviderKind::OpenaiCompat => {
                    let base_url = provider_config
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
                    Provider::OpenAiCompat {
                        base_url,
                        api_key: provider_config.api_key.clone(),
                    }
                }
                ProviderKind::Anthropic => {
                    if provider_config.api_key.is_empty() {
                        return Err(Error::Config(format!(
                            "provider '{}' (anthropic) requires an api_key",
                            route.provider
                        )));
                    }
                    Provider::Anthropic {
                        api_key: provider_config.api_key.clone(),
                    }
                }
                ProviderKind::Google => {
                    if provider_config.api_key.is_empty() {
                        return Err(Error::Config(format!(
                            "provider '{}' (google) requires an api_key",
                            route.provider
                        )));
                    }
                    Provider::Google {
                        api_key: provider_config.api_key.clone(),
                    }
                }
                ProviderKind::Bedrock => {
                    return Err(Error::Config(format!(
                        "provider '{}' (bedrock) is not yet supported",
                        route.provider
                    )));
                }
            };

            providers.insert(model_name.clone(), provider);
        }

        Ok(ProviderRegistry { providers })
    }

    /// Look up the provider for a model name.
    pub fn get(&self, model: &str) -> Option<&Provider> {
        self.providers.get(model)
    }
}

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
    /// Returns an error if a provider is missing a required api_key.
    pub fn from_config(config: &GatewayConfig) -> Result<Self, Error> {
        let mut providers = HashMap::new();

        for (provider_name, provider_config) in &config.providers {
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
                            "provider '{provider_name}' (anthropic) requires an api_key",
                        )));
                    }
                    Provider::Anthropic {
                        api_key: provider_config.api_key.clone(),
                    }
                }
                ProviderKind::Google => {
                    if provider_config.api_key.is_empty() {
                        return Err(Error::Config(format!(
                            "provider '{provider_name}' (google) requires an api_key",
                        )));
                    }
                    Provider::Google {
                        api_key: provider_config.api_key.clone(),
                    }
                }
                ProviderKind::Bedrock => {
                    return Err(Error::Config(format!(
                        "provider '{provider_name}' (bedrock) is not yet supported",
                    )));
                }
            };

            for model_name in &provider_config.models {
                providers.insert(model_name.clone(), provider.clone());
            }
        }

        Ok(ProviderRegistry { providers })
    }

    /// Look up the provider for a model name.
    pub fn get(&self, model: &str) -> Option<&Provider> {
        self.providers.get(model)
    }
}

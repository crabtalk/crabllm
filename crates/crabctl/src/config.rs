use crate::error::Error;
use serde::Deserialize;
use std::path::PathBuf;

const DEFAULT_URL: &str = "http://localhost:8080";

#[derive(Deserialize, Default)]
struct FileConfig {
    url: Option<String>,
    token: Option<String>,
}

pub struct Config {
    pub url: String,
    pub token: String,
}

impl Config {
    /// Resolve config from CLI flags merged with ~/.crabllm/config.toml.
    /// Flags take precedence over file values.
    pub fn resolve(flag_url: Option<String>, flag_token: Option<String>) -> Result<Self, Error> {
        let file = load_file();
        let url = flag_url
            .or(file.url)
            .unwrap_or_else(|| DEFAULT_URL.to_string());
        let token = flag_token.or(file.token).ok_or_else(|| {
            Error::Config(
                "admin token required: pass --token or set 'token' in ~/.crabllm/config.toml"
                    .into(),
            )
        })?;
        Ok(Self { url, token })
    }
}

fn config_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".crabllm").join("config.toml"))
}

fn load_file() -> FileConfig {
    let Some(path) = config_path() else {
        return FileConfig::default();
    };
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return FileConfig::default();
    };
    toml::from_str(&contents).unwrap_or_default()
}

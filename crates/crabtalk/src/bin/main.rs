use clap::Parser;
use crabtalk_core::{Extension, GatewayConfig, MemoryStorage, Storage};
use crabtalk_provider::ProviderRegistry;
use crabtalk_proxy::{
    AppState,
    ext::{
        budget::Budget, cache::Cache, logging::RequestLogger, rate_limit::RateLimit,
        usage::UsageTracker,
    },
};
use std::{path::PathBuf, sync::Arc, time::Duration};

#[derive(Parser)]
#[command(name = "crabtalk", about = "High-performance LLM API gateway")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "crabtalk.toml")]
    config: PathBuf,

    /// Override listen address (e.g. 0.0.0.0:8080)
    #[arg(short, long)]
    bind: Option<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let mut config = match GatewayConfig::from_file(&cli.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load config: {e}");
            std::process::exit(1);
        }
    };

    if let Some(bind) = cli.bind {
        config.listen = bind;
    }

    let registry = match ProviderRegistry::from_config(&config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: failed to build provider registry: {e}");
            std::process::exit(1);
        }
    };

    let storage_kind = config
        .storage
        .as_ref()
        .map(|s| s.kind.as_str())
        .unwrap_or("memory");

    match storage_kind {
        #[cfg(feature = "storage-sqlite")]
        "sqlite" => {
            let path = config
                .storage
                .as_ref()
                .and_then(|s| s.path.as_deref())
                .unwrap_or("crabtalk.db");
            let url = format!("sqlite:{path}?mode=rwc");
            let storage = match crabtalk_core::SqliteStorage::open(&url).await {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    eprintln!("error: failed to open sqlite storage: {e}");
                    std::process::exit(1);
                }
            };
            run(config, registry, storage).await;
        }
        #[cfg(not(feature = "storage-sqlite"))]
        "sqlite" => {
            eprintln!("error: sqlite storage requires the 'storage-sqlite' feature");
            std::process::exit(1);
        }
        _ => {
            let storage = Arc::new(MemoryStorage::new());
            run(config, registry, storage).await;
        }
    }
}

async fn run<S: Storage + 'static>(
    config: GatewayConfig,
    registry: ProviderRegistry,
    storage: Arc<S>,
) {
    let extensions = match build_extensions(&config, storage.clone() as Arc<dyn Storage>) {
        Ok(exts) => exts,
        Err(e) => {
            eprintln!("error: failed to build extensions: {e}");
            std::process::exit(1);
        }
    };

    let ext_count = extensions.len();
    let addr = config.listen.clone();
    let model_count = config.models().len();
    let provider_count = config.providers.len();
    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout);

    let state = AppState {
        registry,
        client: reqwest::Client::new(),
        config,
        extensions: Arc::new(extensions),
        storage,
    };

    let app = crabtalk_proxy::router(state);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: failed to bind to {addr}: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "crabtalk listening on {addr} ({model_count} models, {provider_count} providers, {ext_count} extensions)"
    );

    let server =
        axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(shutdown_timeout));
    if let Err(e) = server.await {
        eprintln!("error: server failed: {e}");
        std::process::exit(1);
    }
}

async fn shutdown_signal(drain_timeout: Duration) {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = sigterm.recv() => {}
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await.ok();

    eprintln!(
        "shutdown signal received, draining connections ({}s timeout)...",
        drain_timeout.as_secs()
    );

    // Force exit after drain timeout.
    tokio::spawn(async move {
        tokio::time::sleep(drain_timeout).await;
        eprintln!("drain timeout exceeded, forcing exit");
        std::process::exit(0);
    });
}

fn build_extensions(
    config: &GatewayConfig,
    storage: Arc<dyn Storage>,
) -> Result<Vec<Box<dyn Extension>>, String> {
    let mut extensions: Vec<Box<dyn Extension>> = Vec::new();
    let mut has_logging = false;

    let ext_table = match &config.extensions {
        Some(toml::Value::Table(t)) => t,
        Some(_) => return Err("[extensions] must be a TOML table".to_string()),
        None => return Ok(extensions),
    };

    for (name, value) in ext_table {
        match name.as_str() {
            "rate_limit" => {
                let ext = RateLimit::new(value, storage.clone())?;
                extensions.push(Box::new(ext));
            }
            "usage" => {
                let ext = UsageTracker::new(value, storage.clone())?;
                extensions.push(Box::new(ext));
            }
            "cache" => {
                let ext = Cache::new(value, storage.clone())?;
                extensions.push(Box::new(ext));
            }
            "budget" => {
                let ext = Budget::new(value, storage.clone(), config.pricing.clone())?;
                extensions.push(Box::new(ext));
            }
            "logging" => {
                let ext = RequestLogger::new(value)?;
                extensions.push(Box::new(ext));
                has_logging = true;
            }
            unknown => {
                return Err(format!(
                    "unknown extension '{unknown}'. valid extensions: rate_limit, usage, cache, budget, logging"
                ));
            }
        }
    }

    // Initialize tracing subscriber if logging extension is enabled.
    if has_logging {
        tracing_subscriber::fmt::init();
    }

    Ok(extensions)
}

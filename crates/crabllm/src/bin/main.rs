use clap::{Parser, Subcommand};
use crabllm_core::{Extension, GatewayConfig, Storage};
use crabllm_provider::ProviderRegistry;
use crabllm_proxy::{
    AppState,
    ext::{
        budget::Budget, cache::Cache, logging::RequestLogger, rate_limit::RateLimit,
        usage::UsageTracker,
    },
    storage::MemoryStorage,
};
use std::{path::PathBuf, sync::Arc, time::Duration};

#[derive(Parser)]
#[command(name = "crabllm", about = "High-performance LLM API gateway")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the API gateway
    Serve {
        /// Path to config file
        #[arg(short, long, default_value = "crabllm.toml")]
        config: PathBuf,

        /// Override listen address (e.g. 0.0.0.0:8080)
        #[arg(short, long)]
        bind: Option<String>,
    },
    /// Manage llama.cpp server and models
    #[command(name = "llamacpp")]
    LlamaCpp {
        #[command(subcommand)]
        action: LlamaCppAction,
    },
}

#[derive(Subcommand)]
enum LlamaCppAction {
    /// Download the llama-server binary for this platform
    Download {
        /// Release tag (e.g. b4567). Defaults to latest.
        #[arg(short, long)]
        tag: Option<String>,
    },
    /// Check that llama-server is installed and reachable
    Check,
    /// Show the resolved path to the llama-server binary
    Which,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::LlamaCpp { action }) => run_llamacpp(action),
        Some(Commands::Serve { config, bind }) => serve(config, bind).await,
        // Default: serve with default config path.
        None => serve(PathBuf::from("crabllm.toml"), None).await,
    }
}

fn run_llamacpp(action: LlamaCppAction) {
    match action {
        LlamaCppAction::Download { tag } => match crabllm_llamacpp::download(tag.as_deref()) {
            Ok(path) => {
                eprintln!("llama-server ready at {}", path.display());
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
        LlamaCppAction::Check => {
            match crabllm_llamacpp::find_server_binary() {
                Ok(path) => {
                    eprintln!("llama-server found: {}", path.display());
                    let output = std::process::Command::new(&path).arg("--version").output();
                    match output {
                        Ok(out) => {
                            // llama-server may print version to stdout or stderr.
                            let version = String::from_utf8_lossy(&out.stdout);
                            let version = version.trim();
                            if !version.is_empty() {
                                eprintln!("{version}");
                            } else {
                                let version = String::from_utf8_lossy(&out.stderr);
                                let version = version.trim();
                                if !version.is_empty() {
                                    eprintln!("{version}");
                                }
                            }
                        }
                        Err(_) => eprintln!("(could not determine version)"),
                    }
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
        LlamaCppAction::Which => match crabllm_llamacpp::find_server_binary() {
            Ok(path) => println!("{}", path.display()),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
    }
}

async fn serve(config_path: PathBuf, bind: Option<String>) {
    let mut config = match GatewayConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load config: {e}");
            std::process::exit(1);
        }
    };

    if let Some(bind) = bind {
        config.listen = bind;
    }

    let (registry, llama_servers) = match ProviderRegistry::from_config(&config) {
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
        #[cfg(feature = "storage-redis")]
        "redis" => {
            let url = config
                .storage
                .as_ref()
                .and_then(|s| s.path.as_deref())
                .unwrap_or("redis://127.0.0.1:6379");
            let storage = match crabllm_proxy::storage::RedisStorage::open(url).await {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    eprintln!("error: failed to open redis storage: {e}");
                    std::process::exit(1);
                }
            };
            run(config, registry, storage, llama_servers).await;
        }
        #[cfg(not(feature = "storage-redis"))]
        "redis" => {
            eprintln!("error: redis storage requires the 'storage-redis' feature");
            std::process::exit(1);
        }
        #[cfg(feature = "storage-sqlite")]
        "sqlite" => {
            let path = config
                .storage
                .as_ref()
                .and_then(|s| s.path.as_deref())
                .unwrap_or("crabllm.db");
            let url = format!("sqlite:{path}?mode=rwc");
            let storage = match crabllm_proxy::storage::SqliteStorage::open(&url).await {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    eprintln!("error: failed to open sqlite storage: {e}");
                    std::process::exit(1);
                }
            };
            run(config, registry, storage, llama_servers).await;
        }
        #[cfg(not(feature = "storage-sqlite"))]
        "sqlite" => {
            eprintln!("error: sqlite storage requires the 'storage-sqlite' feature");
            std::process::exit(1);
        }
        _ => {
            let storage = Arc::new(MemoryStorage::new());
            run(config, registry, storage, llama_servers).await;
        }
    }
}

async fn run<S: Storage + 'static>(
    config: GatewayConfig,
    registry: ProviderRegistry,
    storage: Arc<S>,
    mut llama_servers: Vec<crabllm_provider::LlamaCppServer>,
) {
    let (extensions, admin_routes) =
        match build_extensions(&config, storage.clone() as Arc<dyn Storage>) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("error: failed to build extensions: {e}");
                std::process::exit(1);
            }
        };

    let ext_count = extensions.len();
    let addr = config.listen.clone();
    let model_count = registry.model_names().count();
    let provider_count = config.providers.len();
    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout);

    let key_map = config
        .keys
        .iter()
        .map(|k| (k.key.clone(), k.name.clone()))
        .collect();

    let state = AppState {
        registry,
        client: reqwest::Client::new(),
        config,
        extensions: Arc::new(extensions),
        storage,
        key_map,
    };

    let app = crabllm_proxy::router(state, admin_routes);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: failed to bind to {addr}: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "crabllm listening on {addr} ({model_count} models, {provider_count} providers, {ext_count} extensions)"
    );

    let server =
        axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(shutdown_timeout));
    if let Err(e) = server.await {
        eprintln!("error: server failed: {e}");
    }

    // Explicitly stop llama-server children before exit.
    // This runs after graceful shutdown completes, ensuring destructors
    // fire rather than relying on process::exit which skips them.
    for server in &mut llama_servers {
        server.stop();
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

type Extensions = (Vec<Box<dyn Extension>>, Vec<axum::Router>);

fn build_extensions(
    config: &GatewayConfig,
    storage: Arc<dyn Storage>,
) -> Result<Extensions, String> {
    let mut extensions: Vec<Box<dyn Extension>> = Vec::new();
    let mut admin_routes: Vec<axum::Router> = Vec::new();
    let mut has_logging = false;

    let ext_table = match &config.extensions {
        Some(serde_json::Value::Object(t)) => t,
        Some(_) => return Err("[extensions] must be a table".to_string()),
        None => return Ok((extensions, admin_routes)),
    };

    for (name, value) in ext_table {
        match name.as_str() {
            "rate_limit" => {
                let ext = RateLimit::new(value, storage.clone())?;
                extensions.push(Box::new(ext));
            }
            "usage" => {
                let ext = UsageTracker::new(value, storage.clone())?;
                admin_routes.push(ext.admin_routes());
                extensions.push(Box::new(ext));
            }
            "cache" => {
                let ext = Cache::new(value, storage.clone())?;
                admin_routes.push(ext.admin_routes());
                extensions.push(Box::new(ext));
            }
            "budget" => {
                let ext = Budget::new(value, storage.clone(), config.pricing.clone())?;
                admin_routes.push(ext.admin_routes());
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

    Ok((extensions, admin_routes))
}

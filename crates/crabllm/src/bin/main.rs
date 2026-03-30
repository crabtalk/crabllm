use clap::{Parser, Subcommand};
use crabllm_core::{Extension, GatewayConfig, Storage};
use crabllm_provider::ProviderRegistry;
use crabllm_proxy::{
    AppState,
    ext::{
        audit::AuditLogger, budget::Budget, cache::Cache, logging::RequestLogger,
        rate_limit::RateLimit, usage::UsageTracker,
    },
    storage::MemoryStorage,
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

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
    #[cfg(feature = "llamacpp")]
    #[command(name = "llamacpp")]
    LlamaCpp {
        #[command(subcommand)]
        action: LlamaCppAction,
    },
}

#[cfg(feature = "llamacpp")]
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
        #[cfg(feature = "llamacpp")]
        Some(Commands::LlamaCpp { action }) => run_llamacpp(action),
        Some(Commands::Serve { config, bind }) => serve(config, bind).await,
        // Default: serve with default config path.
        None => serve(PathBuf::from("crabllm.toml"), None).await,
    }
}

#[cfg(feature = "llamacpp")]
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

/// Spawn llama-server processes for all LlamaCpp providers in config.
///
/// For each LlamaCpp provider, this finds the llama-server binary, spawns a
/// child process, waits for it to become healthy, then rewrites the config
/// entry to OpenaiCompat pointing at the local server. The caller must hold
/// the returned handles alive — dropping them kills the child processes.
#[cfg(feature = "llamacpp")]
fn spawn_llamacpp_servers(
    config: &mut GatewayConfig,
) -> Result<Vec<crabllm_llamacpp::LlamaCppServer>, crabllm_core::Error> {
    use crabllm_core::ProviderKind;
    use crabllm_llamacpp::{LlamaCppConfig, LlamaCppServer};

    let has_llamacpp = config
        .providers
        .values()
        .any(|c| c.kind == ProviderKind::LlamaCpp);
    if !has_llamacpp {
        return Ok(Vec::new());
    }

    let bin = crabllm_llamacpp::find_server_binary()?;
    let mut servers = Vec::new();

    for (name, pc) in &mut config.providers {
        if pc.kind != ProviderKind::LlamaCpp {
            continue;
        }

        let model_path = pc.model_path.as_ref().ok_or_else(|| {
            crabllm_core::Error::Config(format!("provider '{name}' (llamacpp) requires model_path"))
        })?;

        eprintln!("starting llama-server for provider '{name}' (model: {model_path})");

        let llama_config = LlamaCppConfig {
            model_path: PathBuf::from(model_path),
            n_gpu_layers: pc.n_gpu_layers.unwrap_or(0),
            n_ctx: pc.n_ctx.unwrap_or(2048),
            n_threads: pc.n_threads,
        };

        let server = LlamaCppServer::spawn(&bin, &llama_config).map_err(|e| {
            crabllm_core::Error::Config(format!(
                "provider '{name}': failed to start llama-server: {e}"
            ))
        })?;

        eprintln!(
            "llama-server for provider '{name}' ready on port {}",
            server.port()
        );

        // Rewrite config entry so the provider crate sees OpenaiCompat.
        pc.kind = ProviderKind::OpenaiCompat;
        pc.base_url = Some(server.base_url());

        servers.push(server);
    }

    Ok(servers)
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

    // Spawn llama-server processes and rewrite their config entries to
    // OpenaiCompat before building the registry. Held on this stack frame
    // for lifetime — Drop kills the child processes after run() returns.
    #[cfg(feature = "llamacpp")]
    let _llama_servers = match spawn_llamacpp_servers(&mut config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to start llama-server: {e}");
            std::process::exit(1);
        }
    };

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
            run(config, registry, storage).await;
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
    let (extensions, mut admin_routes) =
        match build_extensions(&config, storage.clone() as Arc<dyn Storage>) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("error: failed to build extensions: {e}");
                std::process::exit(1);
            }
        };

    // Install Prometheus metrics recorder and expose /metrics endpoint.
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install metrics recorder");
    admin_routes.push(axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || async move { handle.render() }),
    ));

    let ext_count = extensions.len();
    let addr = config.listen.clone();
    let model_count = registry.model_names().count();
    let provider_count = config.providers.len();
    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout);

    // Build key_map from TOML config keys.
    let key_map: HashMap<String, String> = config
        .keys
        .iter()
        .map(|k| (k.key.clone(), k.name.clone()))
        .collect();
    let key_map = Arc::new(RwLock::new(key_map));

    // Load stored keys and merge (TOML takes precedence on conflicts).
    crabllm_proxy::admin::load_stored_keys(
        storage.as_ref() as &dyn crabllm_core::Storage,
        &config.keys,
        &key_map,
    )
    .await;

    // Enable admin key management if admin_token is configured.
    if let Some(ref admin_token) = config.admin_token {
        admin_routes.push(crabllm_proxy::admin::key_admin_routes(
            storage.clone() as Arc<dyn crabllm_core::Storage>,
            key_map.clone(),
            admin_token.clone(),
            config.keys.clone(),
        ));
    }

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
            "audit" => {
                let ext = AuditLogger::new(value, storage.clone(), config.pricing.clone())?;
                admin_routes.push(ext.admin_routes());
                extensions.push(Box::new(ext));
            }
            unknown => {
                return Err(format!(
                    "unknown extension '{unknown}'. valid extensions: rate_limit, usage, cache, budget, logging, audit"
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

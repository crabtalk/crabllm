use bytes::Bytes;
use clap::{Parser, Subcommand};
use crabllm_core::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, Extension, GatewayConfig,
    ImageRequest, MultipartField, Provider, Storage,
};
#[cfg(feature = "llamacpp")]
use crabllm_llamacpp::LlamaCppProvider;
use crabllm_provider::{ProviderRegistry, RemoteProvider};
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

/// Concrete provider type the gateway binary composes.
///
/// Union of every provider source the binary links — remote HTTP APIs
/// via [`RemoteProvider`] and (behind `--features llamacpp`) a local
/// llama.cpp backend via [`LlamaCppProvider`]. The proxy crate is
/// generic over `P: Provider`; the binary picks this enum as `P` so
/// dispatch monomorphizes through a match/delegate with no dyn or
/// per-call boxing.
///
/// `LlamaCpp` is feature-gated so a no-llamacpp build collapses to a
/// single-variant enum with zero runtime cost. The trait methods live
/// inline in this file because extracting them into their own file
/// would require a sibling module under `src/bin/`, which is overkill
/// for ~70 lines of match-delegate.
enum Dispatch {
    Remote(RemoteProvider),
    #[cfg(feature = "llamacpp")]
    LlamaCpp(LlamaCppProvider),
}

impl Provider for Dispatch {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        match self {
            Self::Remote(p) => p.chat_completion(request).await,
            #[cfg(feature = "llamacpp")]
            Self::LlamaCpp(p) => p.chat_completion(request).await,
        }
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        match self {
            Self::Remote(p) => p.chat_completion_stream(request).await,
            #[cfg(feature = "llamacpp")]
            Self::LlamaCpp(p) => p.chat_completion_stream(request).await,
        }
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        match self {
            Self::Remote(p) => p.embedding(request).await,
            #[cfg(feature = "llamacpp")]
            Self::LlamaCpp(p) => p.embedding(request).await,
        }
    }

    async fn image_generation(&self, request: &ImageRequest) -> Result<(Bytes, String), Error> {
        match self {
            Self::Remote(p) => p.image_generation(request).await,
            #[cfg(feature = "llamacpp")]
            Self::LlamaCpp(p) => p.image_generation(request).await,
        }
    }

    async fn audio_speech(&self, request: &AudioSpeechRequest) -> Result<(Bytes, String), Error> {
        match self {
            Self::Remote(p) => p.audio_speech(request).await,
            #[cfg(feature = "llamacpp")]
            Self::LlamaCpp(p) => p.audio_speech(request).await,
        }
    }

    async fn audio_transcription(
        &self,
        model: &str,
        fields: &[MultipartField],
    ) -> Result<(Bytes, String), Error> {
        match self {
            Self::Remote(p) => p.audio_transcription(model, fields).await,
            #[cfg(feature = "llamacpp")]
            Self::LlamaCpp(p) => p.audio_transcription(model, fields).await,
        }
    }
}

/// Attach the llama.cpp local backend to the registry and return the
/// pool so `serve()` can hold it alive and drive a clean shutdown.
///
/// The pool is constructed with the configured knobs, starts its idle
/// monitor, and is wrapped in a single `LlamaCppProvider` that's
/// cloned into one `Deployment` per configured model — all sharing the
/// same pool. A startup resolution check walks every model and fails
/// fast if its GGUF is neither cached nor a valid path, so missing
/// models produce a clear error at boot instead of on the first chat
/// completion.
#[cfg(feature = "llamacpp")]
fn wire_llamacpp(
    registry: &mut ProviderRegistry<Dispatch>,
    cfg: &crabllm_core::LlamaCppGatewayConfig,
) -> Result<Arc<crabllm_llamacpp::ServerPool>, crabllm_core::Error> {
    use crabllm_llamacpp::{ServerPool, registry as reg};
    use crabllm_provider::Deployment;
    use std::path::Path;

    let bin = crabllm_llamacpp::find_server_binary()?;

    let cache_dir = match &cfg.cache_dir {
        Some(s) => PathBuf::from(s),
        None => reg::default_cache_dir()?,
    };

    // Resolution check — fail fast on missing models without spawning.
    for model in &cfg.models {
        if reg::cached_model_path(model, &cache_dir).is_some() {
            continue;
        }
        if Path::new(model).exists() {
            continue;
        }
        let (name, tag) = reg::parse_model_name(model);
        return Err(crabllm_core::Error::Config(format!(
            "llamacpp model '{name}:{tag}' not cached. Run: crabllm-llamacpp pull {model}"
        )));
    }

    let mut pool = ServerPool::new(bin, cache_dir);
    if let Some(secs) = cfg.idle_timeout_secs {
        pool = pool.with_idle_timeout(Duration::from_secs(secs));
    }
    if let Some(n) = cfg.n_gpu_layers {
        pool = pool.with_gpu_layers(n);
    }
    if let Some(n) = cfg.n_ctx {
        pool = pool.with_ctx_size(n);
    }
    if let Some(n) = cfg.n_threads {
        pool = pool.with_threads(n);
    }
    let pool = Arc::new(pool);
    pool.start_idle_monitor();

    let provider = LlamaCppProvider::new(pool.clone(), crabllm_provider::make_client());

    for model in &cfg.models {
        let deployment = Deployment {
            provider: Dispatch::LlamaCpp(provider.clone()),
            weight: 1,
            // Retry against a local child is pointless; the pool already
            // serializes starts and the child is process-local.
            max_retries: 0,
            // Generation can legitimately run long on CPU.
            timeout: Duration::from_secs(600),
        };
        registry.insert_deployment(model.clone(), "llamacpp".to_string(), deployment);
    }

    Ok(pool)
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

    #[cfg_attr(not(feature = "llamacpp"), expect(unused_mut))]
    let mut registry: ProviderRegistry<Dispatch> =
        match ProviderRegistry::from_config(&config, Dispatch::Remote) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: failed to build provider registry: {e}");
                std::process::exit(1);
            }
        };

    // Wire the llama.cpp local backend if [llamacpp] is present. The
    // returned pool is held on this stack frame so the child processes
    // stay alive for the server's lifetime, then explicitly stopped
    // after `run(...)` returns.
    #[cfg(feature = "llamacpp")]
    let llama_pool = match config.llamacpp.clone() {
        Some(cfg) => match wire_llamacpp(&mut registry, &cfg) {
            Ok(pool) => Some(pool),
            Err(e) => {
                eprintln!("error: failed to wire llamacpp: {e}");
                std::process::exit(1);
            }
        },
        None => None,
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

    // Stop llama-server child processes cleanly after the gateway has
    // drained. `LlamaCppServer::Drop` also kills the process, but
    // calling `stop_all` here gives the idle monitor a chance to see
    // the shutdown flag and exit without another tick.
    #[cfg(feature = "llamacpp")]
    if let Some(pool) = llama_pool {
        pool.stop_all().await;
    }
}

async fn run<S: Storage + 'static>(
    config: GatewayConfig,
    registry: ProviderRegistry<Dispatch>,
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
    let provider_count = registry.provider_count();
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

    let state: AppState<S, Dispatch> = AppState {
        registry,
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

    let server = axum::serve(NoDelayListener(listener), app)
        .with_graceful_shutdown(shutdown_signal(shutdown_timeout));
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

/// TCP listener that sets TCP_NODELAY on every accepted connection.
///
/// Without this, Nagle's algorithm buffers small SSE writes and waits for
/// ACKs (~40 ms delayed-ACK window), turning sub-ms streaming overhead into
/// ~44 ms per request and holding connections open far longer than necessary.
struct NoDelayListener(tokio::net::TcpListener);

impl axum::serve::Listener for NoDelayListener {
    type Io = tokio::net::TcpStream;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.0.accept().await {
                Ok((stream, addr)) => {
                    let _ = stream.set_nodelay(true);
                    return (stream, addr);
                }
                Err(_) => continue,
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.0.local_addr()
    }
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

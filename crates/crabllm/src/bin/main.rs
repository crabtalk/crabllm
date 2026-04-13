use arc_swap::ArcSwap;
use bytes::Bytes;
use clap::{Parser, Subcommand};
use crabllm_core::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, Extension, GatewayConfig,
    ImageRequest, MultipartField, Provider, Storage,
};
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
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Serve { config, bind }) => serve(config, bind).await,
        // Default: serve with default config path.
        None => serve(PathBuf::from("crabllm.toml"), None).await,
    }
}

/// Concrete provider type the gateway binary composes.
///
/// The proxy crate is generic over `P: Provider`; the binary picks
/// this enum as `P` so dispatch monomorphizes through a match/delegate
/// with no dyn or per-call boxing. Currently a single-variant wrapper
/// around [`RemoteProvider`] — local backends (MLX, llama.cpp) are
/// separate binaries, not compiled into the gateway.
enum Dispatch {
    Remote(RemoteProvider),
}

impl Provider for Dispatch {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        match self {
            Self::Remote(p) => p.chat_completion(request).await,
        }
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        match self {
            Self::Remote(p) => p.chat_completion_stream(request).await,
        }
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        match self {
            Self::Remote(p) => p.embedding(request).await,
        }
    }

    async fn image_generation(&self, request: &ImageRequest) -> Result<(Bytes, String), Error> {
        match self {
            Self::Remote(p) => p.image_generation(request).await,
        }
    }

    async fn audio_speech(&self, request: &AudioSpeechRequest) -> Result<(Bytes, String), Error> {
        match self {
            Self::Remote(p) => p.audio_speech(request).await,
        }
    }

    async fn audio_transcription(
        &self,
        model: &str,
        fields: &[MultipartField],
    ) -> Result<(Bytes, String), Error> {
        match self {
            Self::Remote(p) => p.audio_transcription(model, fields).await,
        }
    }

    fn is_openai_compat(&self) -> bool {
        match self {
            Self::Remote(p) => p.is_openai_compat(),
        }
    }

    fn is_anthropic_compat(&self) -> bool {
        match self {
            Self::Remote(p) => p.is_anthropic_compat(),
        }
    }

    async fn chat_completion_raw(&self, model: &str, raw_body: Bytes) -> Result<Bytes, Error> {
        match self {
            Self::Remote(p) => p.chat_completion_raw(model, raw_body).await,
        }
    }

    async fn anthropic_messages_raw(&self, raw_body: Bytes) -> Result<Bytes, Error> {
        match self {
            Self::Remote(p) => p.anthropic_messages_raw(raw_body).await,
        }
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

    let registry: ProviderRegistry<Dispatch> =
        match ProviderRegistry::from_config(&config, Dispatch::Remote) {
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
            run(config, config_path.clone(), registry, storage).await;
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
            run(config, config_path.clone(), registry, storage).await;
        }
        #[cfg(not(feature = "storage-sqlite"))]
        "sqlite" => {
            eprintln!("error: sqlite storage requires the 'storage-sqlite' feature");
            std::process::exit(1);
        }
        _ => {
            let storage = Arc::new(MemoryStorage::new());
            run(config, config_path.clone(), registry, storage).await;
        }
    }
}

async fn run<S: Storage + 'static>(
    config: GatewayConfig,
    config_path: PathBuf,
    registry: ProviderRegistry<Dispatch>,
    storage: Arc<S>,
) {
    // Initialize model metadata overrides from storage.
    let model_overrides = Arc::new(RwLock::new(HashMap::new()));
    crabllm_proxy::admin_models::load_stored_models(
        storage.as_ref() as &dyn crabllm_core::Storage,
        &model_overrides,
    )
    .await;

    let (extensions, mut admin_routes) = match build_extensions(
        &config,
        storage.clone() as Arc<dyn Storage>,
        model_overrides.clone(),
    ) {
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
    let registry = Arc::new(ArcSwap::from_pointee(registry));

    // Build key_map from TOML config keys.
    let key_map: HashMap<String, String> = config
        .keys
        .iter()
        .map(|k| (k.key.clone(), k.name.clone()))
        .collect();
    let key_map = Arc::new(RwLock::new(key_map));

    // Persist TOML key configs to storage so extensions (rate limiter,
    // etc.) can look up per-key config without a separate in-memory map.
    // Overwrites on every startup to reflect config edits.
    for kc in &config.keys {
        let skey = crabllm_core::storage_key(&crabllm_proxy::PREFIX_KEYS, kc.name.as_bytes());
        if let Ok(value) = serde_json::to_vec(kc) {
            let _ = storage.set(&skey, value).await;
        }
    }

    // Load stored keys and merge (TOML takes precedence on conflicts).
    crabllm_proxy::admin::load_stored_keys(
        storage.as_ref() as &dyn crabllm_core::Storage,
        &config.keys,
        &key_map,
    )
    .await;

    // Enable admin APIs if admin_token is configured.
    if let Some(ref admin_token) = config.admin_token {
        admin_routes.push(crabllm_proxy::admin::key_admin_routes(
            storage.clone() as Arc<dyn crabllm_core::Storage>,
            key_map.clone(),
            admin_token.clone(),
            config.keys.clone(),
        ));
        admin_routes.push(crabllm_proxy::admin_models::model_admin_routes(
            storage.clone() as Arc<dyn crabllm_core::Storage>,
            model_overrides.clone(),
            config.clone(),
            config_path.clone(),
            admin_token.clone(),
        ));
        let rebuilder: crabllm_proxy::admin_providers::Rebuilder<Dispatch> =
            Arc::new(|config: &GatewayConfig| {
                ProviderRegistry::from_config(config, Dispatch::Remote)
            });
        admin_routes.push(crabllm_proxy::admin_providers::provider_admin_routes(
            registry.clone(),
            config_path,
            admin_token.clone(),
            rebuilder,
        ));
    }

    let state: AppState<S, Dispatch> = AppState {
        registry,
        config,
        extensions: Arc::new(extensions),
        storage,
        key_map,
        model_overrides,
        usage_events: None,
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
    model_overrides: Arc<RwLock<HashMap<String, crabllm_core::ModelInfo>>>,
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
                let ext = Budget::new(
                    value,
                    storage.clone(),
                    config.models.clone(),
                    model_overrides.clone(),
                )?;
                admin_routes.push(ext.admin_routes());
                extensions.push(Box::new(ext));
            }
            "logging" => {
                let ext = RequestLogger::new(value)?;
                extensions.push(Box::new(ext));
                has_logging = true;
            }
            "audit" => {
                let ext = AuditLogger::new(
                    value,
                    storage.clone(),
                    config.models.clone(),
                    model_overrides.clone(),
                )?;
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

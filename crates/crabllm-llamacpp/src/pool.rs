use crate::{
    registry::{self, cached_model_path},
    server::{LlamaCppConfig, LlamaCppServer},
};
use crabllm_core::Error;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, Notify};

/// Default idle timeout: 30 minutes.
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// How often the idle monitor checks for expired servers.
const MONITOR_INTERVAL: Duration = Duration::from_secs(60);
/// Default GPU layers: 999 lets llama.cpp auto-limit to actual model layers.
const DEFAULT_GPU_LAYERS: u32 = 999;
/// Default context size in tokens.
const DEFAULT_CTX_SIZE: u32 = 4096;

/// Per-model state in the pool.
enum ModelState {
    /// Server is starting — other callers wait on the notify.
    Starting { notify: Arc<Notify> },
    /// Server is running and ready to serve requests.
    Running {
        server: LlamaCppServer,
        last_request: Instant,
    },
}

/// Manages a pool of on-demand llama-server processes.
///
/// Each model gets its own llama-server subprocess. Servers start on first
/// request and stop after the idle timeout. The pool resolves model names
/// to cached GGUF files (fetched via the Ollama registry).
pub struct ServerPool {
    models: Mutex<HashMap<String, ModelState>>,
    bin: PathBuf,
    cache_dir: PathBuf,
    idle_timeout: Duration,
    shutdown: AtomicBool,
    n_gpu_layers: u32,
    n_ctx: u32,
    n_threads: Option<u32>,
}

impl std::fmt::Debug for ServerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerPool")
            .field("cache_dir", &self.cache_dir)
            .field("idle_timeout", &self.idle_timeout)
            .finish_non_exhaustive()
    }
}

impl ServerPool {
    /// Create a new server pool.
    ///
    /// `bin` is the path to the llama-server binary.
    /// `cache_dir` is where GGUF model files are cached.
    pub fn new(bin: PathBuf, cache_dir: PathBuf) -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
            bin,
            cache_dir,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            shutdown: AtomicBool::new(false),
            n_gpu_layers: DEFAULT_GPU_LAYERS,
            n_ctx: DEFAULT_CTX_SIZE,
            n_threads: None,
        }
    }

    /// Override the idle timeout (default: 30 minutes).
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Override GPU layers (default: 999 = auto).
    pub fn with_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = n;
        self
    }

    /// Override context size (default: 4096).
    pub fn with_ctx_size(mut self, n: u32) -> Self {
        self.n_ctx = n;
        self
    }

    /// Override thread count (default: system-chosen).
    pub fn with_threads(mut self, n: u32) -> Self {
        self.n_threads = Some(n);
        self
    }

    /// Ensure a model's llama-server is running, starting it if needed.
    ///
    /// Returns the OpenAI-compatible base URL (e.g., `http://127.0.0.1:{port}/v1`).
    /// Serializes per-model: concurrent requests for the same model wait
    /// rather than spawning duplicate servers.
    pub async fn ensure_running(&self, model: &str) -> Result<String, Error> {
        loop {
            let notify = {
                let mut models = self.models.lock().await;
                match models.get_mut(model) {
                    Some(ModelState::Running {
                        last_request,
                        server,
                    }) => {
                        *last_request = Instant::now();
                        return Ok(server.base_url());
                    }
                    Some(ModelState::Starting { notify }) => {
                        // Another caller is starting this model — wait for it.
                        Arc::clone(notify)
                    }
                    None => {
                        if self.shutdown.load(Ordering::Relaxed) {
                            return Err(Error::Internal(
                                "server pool is shutting down".to_string(),
                            ));
                        }
                        // Claim the slot so other callers wait on us.
                        let notify = Arc::new(Notify::new());
                        models.insert(
                            model.to_string(),
                            ModelState::Starting {
                                notify: Arc::clone(&notify),
                            },
                        );
                        // Break out to do the slow spawn work.
                        return self.spawn_server(model, notify).await;
                    }
                }
            };
            // Wait outside the lock for the starter to finish, then re-check.
            notify.notified().await;
        }
    }

    /// Spawn a llama-server for a model. Called after inserting `Starting` state.
    async fn spawn_server(&self, model: &str, notify: Arc<Notify>) -> Result<String, Error> {
        let result = self.try_spawn_server(model).await;

        let mut models = self.models.lock().await;
        match result {
            Ok(server) => {
                let base_url = server.base_url();
                models.insert(
                    model.to_string(),
                    ModelState::Running {
                        server,
                        last_request: Instant::now(),
                    },
                );
                notify.notify_waiters();
                Ok(base_url)
            }
            Err(e) => {
                models.remove(model);
                notify.notify_waiters();
                Err(e)
            }
        }
    }

    /// Do the actual blocking spawn work.
    async fn try_spawn_server(&self, model: &str) -> Result<LlamaCppServer, Error> {
        let model_path = self.resolve_model(model)?;
        let config = LlamaCppConfig {
            model_path,
            n_gpu_layers: self.n_gpu_layers,
            n_ctx: self.n_ctx,
            n_threads: self.n_threads,
        };
        let bin = self.bin.clone();
        tokio::task::spawn_blocking(move || LlamaCppServer::spawn(&bin, &config))
            .await
            .map_err(|e| Error::Internal(format!("spawn task failed: {e}")))?
    }

    /// Stop all running servers and prevent new ones from starting.
    pub async fn stop_all(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let to_drop = {
            let mut models = self.models.lock().await;
            // Extract all Running servers. Starting entries get left — their
            // spawners will see the shutdown flag and clean up.
            let running: Vec<String> = models
                .iter()
                .filter(|(_, s)| matches!(s, ModelState::Running { .. }))
                .map(|(k, _)| k.clone())
                .collect();
            let mut servers = Vec::new();
            for key in running {
                if let Some(ModelState::Running { server, .. }) = models.remove(&key) {
                    servers.push(server);
                }
            }
            servers
        };
        // Drop servers outside the lock, in a blocking context.
        if !to_drop.is_empty() {
            tokio::task::spawn_blocking(move || drop(to_drop))
                .await
                .ok();
        }
    }

    /// Spawn a background task that periodically stops idle servers.
    /// Returns a handle that aborts the monitor when dropped.
    pub fn start_idle_monitor(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let pool = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(MONITOR_INTERVAL).await;
                if pool.shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let to_drop = {
                    let mut models = pool.models.lock().await;
                    let expired: Vec<String> = models
                        .iter()
                        .filter(|(_, s)| matches!(s, ModelState::Running { last_request, .. } if last_request.elapsed() > pool.idle_timeout))
                        .map(|(name, _)| name.clone())
                        .collect();
                    let mut servers = Vec::new();
                    for name in &expired {
                        tracing::info!(model = %name, "stopping idle llama-server");
                        if let Some(ModelState::Running { server, .. }) = models.remove(name) {
                            servers.push(server);
                        }
                    }
                    servers
                };
                // Drop servers outside the lock, in a blocking context.
                if !to_drop.is_empty() {
                    tokio::task::spawn_blocking(move || drop(to_drop))
                        .await
                        .ok();
                }
            }
        })
    }

    /// Resolve a model name to a cached GGUF path.
    fn resolve_model(&self, model: &str) -> Result<PathBuf, Error> {
        if let Some(path) = cached_model_path(model, &self.cache_dir) {
            return Ok(path);
        }

        let as_path = Path::new(model);
        if as_path.exists() {
            return Ok(as_path.to_path_buf());
        }

        let (name, tag) = registry::parse_model_name(model);
        Err(Error::Internal(format!(
            "model '{name}:{tag}' not cached. Run: crabllm registry pull {model}"
        )))
    }
}

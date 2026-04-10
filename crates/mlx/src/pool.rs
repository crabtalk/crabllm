//! `MlxPool` — on-demand multi-model cache with idle eviction.
//!
//! Mirrors the shape of `crabllm-llamacpp`'s `ServerPool`: a
//! `HashMap<String, ModelState>` keyed by model name, with
//! `Starting`/`Ready` states so concurrent callers for the same model
//! wait on one loader instead of racing.
//!
//! Idle eviction runs as a detached background task that wakes every
//! `MONITOR_INTERVAL` and drops any model whose `last_used` is older
//! than `idle_timeout`. Dropping an `MlxModel` releases the underlying
//! `Session` — Swift frees the GPU memory.

use crate::model::MlxModel;
use crabllm_core::Error;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, Notify};

const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const MONITOR_INTERVAL: Duration = Duration::from_secs(60);

enum ModelState {
    Starting { notify: Arc<Notify> },
    Ready { model: MlxModel, last_used: Instant },
}

/// Multi-model manager. Cheap to clone internally via `Arc`.
pub struct MlxPool {
    models: Mutex<HashMap<String, ModelState>>,
    idle_timeout: Duration,
    shutdown: AtomicBool,
}

impl std::fmt::Debug for MlxPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxPool")
            .field("idle_timeout", &self.idle_timeout)
            .finish_non_exhaustive()
    }
}

impl MlxPool {
    pub fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            shutdown: AtomicBool::new(false),
        }
    }

    /// Override the idle timeout (default: 30 minutes).
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Ensure a model is loaded and return a cheap clone.
    ///
    /// Per-model serialization: the first caller for a given name
    /// transitions the slot to `Starting` and kicks off
    /// [`MlxModel::new`]; concurrent callers for the same name wait
    /// on a shared `Notify` and re-read the slot afterwards.
    pub async fn ensure_loaded(&self, model_id: &str) -> Result<MlxModel, Error> {
        loop {
            let notify = {
                let mut models = self.models.lock().await;
                match models.get_mut(model_id) {
                    Some(ModelState::Ready { model, last_used }) => {
                        *last_used = Instant::now();
                        return Ok(model.clone());
                    }
                    Some(ModelState::Starting { notify }) => Arc::clone(notify),
                    None => {
                        if self.shutdown.load(Ordering::Relaxed) {
                            return Err(Error::Internal("mlx pool is shutting down".to_string()));
                        }
                        let notify = Arc::new(Notify::new());
                        models.insert(
                            model_id.to_string(),
                            ModelState::Starting {
                                notify: Arc::clone(&notify),
                            },
                        );
                        drop(models);
                        return self.load_model(model_id, notify).await;
                    }
                }
            };
            notify.notified().await;
        }
    }

    async fn load_model(&self, model_id: &str, notify: Arc<Notify>) -> Result<MlxModel, Error> {
        let result = MlxModel::new(model_id).await;
        let mut models = self.models.lock().await;
        match result {
            Ok(model) => {
                models.insert(
                    model_id.to_string(),
                    ModelState::Ready {
                        model: model.clone(),
                        last_used: Instant::now(),
                    },
                );
                notify.notify_waiters();
                Ok(model)
            }
            Err(e) => {
                models.remove(model_id);
                notify.notify_waiters();
                Err(e)
            }
        }
    }

    /// Evict a single model if it's `Ready`. Callers who want more
    /// control over lifecycle can wire this up themselves instead of
    /// relying on the idle monitor.
    pub async fn evict(&self, model_id: &str) {
        let mut models = self.models.lock().await;
        if let Some(ModelState::Ready { .. }) = models.get(model_id) {
            models.remove(model_id);
        }
    }

    /// Drop every `Ready` model and prevent new loads.
    pub async fn stop_all(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let mut models = self.models.lock().await;
        let keys: Vec<String> = models
            .iter()
            .filter(|(_, state)| matches!(state, ModelState::Ready { .. }))
            .map(|(k, _)| k.clone())
            .collect();
        for key in keys {
            models.remove(&key);
        }
    }

    /// Spawn a background task that periodically evicts idle models.
    /// The task exits after `stop_all` is called.
    pub fn start_idle_monitor(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let pool = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(MONITOR_INTERVAL).await;
                if pool.shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let expired: Vec<String> = {
                    let models = pool.models.lock().await;
                    models
                        .iter()
                        .filter_map(|(name, state)| match state {
                            ModelState::Ready { last_used, .. }
                                if last_used.elapsed() > pool.idle_timeout =>
                            {
                                Some(name.clone())
                            }
                            _ => None,
                        })
                        .collect()
                };
                if !expired.is_empty() {
                    let mut models = pool.models.lock().await;
                    for name in &expired {
                        tracing::info!(model = %name, "mlx: evicting idle model");
                        models.remove(name);
                    }
                }
            }
        })
    }
}

impl Default for MlxPool {
    fn default() -> Self {
        Self::new()
    }
}

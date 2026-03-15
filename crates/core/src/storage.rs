use crate::Error;
use dashmap::DashMap;
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicI64, Ordering},
};

/// Fixed-length prefix size in bytes for storage keys.
pub const PREFIX_LEN: usize = 4;

/// A fixed-length prefix that namespaces storage keys per extension.
///
/// The first `PREFIX_LEN` bytes of every key identify which extension
/// owns the data. The remaining bytes are the extension-specific key.
pub type Prefix = [u8; PREFIX_LEN];

/// Key-value pairs returned by `Storage::list`.
pub type KvPairs = Vec<(Vec<u8>, Vec<u8>)>;

/// A pinned, boxed, Send future. Used for dyn-compatible async trait methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Build a full storage key from a fixed-length prefix and a suffix.
pub fn storage_key(prefix: &Prefix, suffix: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(PREFIX_LEN + suffix.len());
    key.extend_from_slice(prefix);
    key.extend_from_slice(suffix);
    key
}

/// Generic async key-value storage backend for extensions.
///
/// Keys are raw bytes. The first `PREFIX_LEN` bytes are a fixed-width
/// namespace prefix. All methods take `&self` and are dyn-compatible
/// via `BoxFuture`. Implementations must be `Send + Sync`.
pub trait Storage: Send + Sync {
    /// Get a value by key. Returns `None` if the key does not exist.
    fn get(&self, key: &[u8]) -> BoxFuture<'_, Result<Option<Vec<u8>>, Error>>;

    /// Set a key to a value, overwriting any existing value.
    fn set(&self, key: &[u8], value: Vec<u8>) -> BoxFuture<'_, Result<(), Error>>;

    /// Atomically increment a counter by `delta`, returning the new value.
    /// If the key does not exist, it is created with an initial value of `delta`.
    fn increment(&self, key: &[u8], delta: i64) -> BoxFuture<'_, Result<i64, Error>>;

    /// List all key-value pairs whose keys start with `prefix`.
    /// Returned in arbitrary order.
    fn list(&self, prefix: &Prefix) -> BoxFuture<'_, Result<KvPairs, Error>>;

    /// Delete a key. No-op if the key does not exist.
    fn delete(&self, key: &[u8]) -> BoxFuture<'_, Result<(), Error>>;
}

/// In-memory storage backend using `DashMap`.
pub struct MemoryStorage {
    data: DashMap<Vec<u8>, Vec<u8>>,
    counters: DashMap<Vec<u8>, AtomicI64>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            data: DashMap::new(),
            counters: DashMap::new(),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for MemoryStorage {
    fn get(&self, key: &[u8]) -> BoxFuture<'_, Result<Option<Vec<u8>>, Error>> {
        let result = self.data.get(key).map(|v| v.value().clone());
        Box::pin(async move { Ok(result) })
    }

    fn set(&self, key: &[u8], value: Vec<u8>) -> BoxFuture<'_, Result<(), Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            self.data.insert(key, value);
            Ok(())
        })
    }

    fn increment(&self, key: &[u8], delta: i64) -> BoxFuture<'_, Result<i64, Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let entry = self
                .counters
                .entry(key)
                .or_insert_with(|| AtomicI64::new(0));
            let new_val = entry.value().fetch_add(delta, Ordering::Relaxed) + delta;
            Ok(new_val)
        })
    }

    fn list(&self, prefix: &Prefix) -> BoxFuture<'_, Result<KvPairs, Error>> {
        let prefix = *prefix;
        Box::pin(async move {
            let mut result = Vec::new();
            for entry in self.data.iter() {
                if entry.key().starts_with(&prefix) {
                    result.push((entry.key().clone(), entry.value().clone()));
                }
            }
            Ok(result)
        })
    }

    fn delete(&self, key: &[u8]) -> BoxFuture<'_, Result<(), Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            self.data.remove(&key);
            self.counters.remove(&key);
            Ok(())
        })
    }
}

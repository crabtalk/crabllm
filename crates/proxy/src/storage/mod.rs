use crabtalk_core::{BoxFuture, Error, KvPairs, Prefix, Storage};
use dashmap::DashMap;
use std::sync::atomic::{AtomicI64, Ordering};

#[cfg(feature = "storage-redis")]
mod redis;
#[cfg(feature = "storage-sqlite")]
mod sqlite;

#[cfg(feature = "storage-redis")]
pub use self::redis::RedisStorage;
#[cfg(feature = "storage-sqlite")]
pub use self::sqlite::SqliteStorage;

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
            // Also include counter keys (used by rate_limit, usage, budget).
            for entry in self.counters.iter() {
                if entry.key().starts_with(&prefix) {
                    let val = entry.value().load(Ordering::Relaxed);
                    result.push((entry.key().clone(), val.to_le_bytes().to_vec()));
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

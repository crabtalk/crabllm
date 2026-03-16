use crate::{BoxFuture, Error, KvPairs, Prefix, Storage};
use sqlx::{Row, SqlitePool, sqlite::SqlitePoolOptions};

pub struct SqliteStorage {
    pool: SqlitePool,
}

impl SqliteStorage {
    pub async fn open(url: &str) -> Result<Self, Error> {
        let pool = SqlitePoolOptions::new()
            .connect(url)
            .await
            .map_err(|e| Error::Internal(format!("sqlite open: {e}")))?;

        sqlx::query("CREATE TABLE IF NOT EXISTS kv (key BLOB PRIMARY KEY, value BLOB NOT NULL)")
            .execute(&pool)
            .await
            .map_err(|e| Error::Internal(format!("sqlite init: {e}")))?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS counters (key BLOB PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0)",
        )
        .execute(&pool)
        .await
        .map_err(|e| Error::Internal(format!("sqlite init: {e}")))?;

        Ok(Self { pool })
    }
}

impl Storage for SqliteStorage {
    fn get(&self, key: &[u8]) -> BoxFuture<'_, Result<Option<Vec<u8>>, Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let row = sqlx::query("SELECT value FROM kv WHERE key = ?")
                .bind(&key)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(row.map(|r| r.get::<Vec<u8>, _>("value")))
        })
    }

    fn set(&self, key: &[u8], value: Vec<u8>) -> BoxFuture<'_, Result<(), Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            sqlx::query(
                "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            )
            .bind(&key)
            .bind(&value)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(())
        })
    }

    fn increment(&self, key: &[u8], delta: i64) -> BoxFuture<'_, Result<i64, Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let row = sqlx::query(
                "INSERT INTO counters (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = value + excluded.value RETURNING value",
            )
            .bind(&key)
            .bind(delta)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(row.get::<i64, _>("value"))
        })
    }

    fn list(&self, prefix: &Prefix) -> BoxFuture<'_, Result<KvPairs, Error>> {
        let prefix_vec = prefix.to_vec();
        let mut upper = prefix_vec.clone();
        if let Some(last) = upper.last_mut() {
            *last = last.wrapping_add(1);
        }

        Box::pin(async move {
            let mut result = Vec::new();

            let kv_rows = sqlx::query("SELECT key, value FROM kv WHERE key >= ? AND key < ?")
                .bind(&prefix_vec)
                .bind(&upper)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;

            for row in &kv_rows {
                result.push((row.get::<Vec<u8>, _>("key"), row.get::<Vec<u8>, _>("value")));
            }

            let counter_rows =
                sqlx::query("SELECT key, value FROM counters WHERE key >= ? AND key < ?")
                    .bind(&prefix_vec)
                    .bind(&upper)
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| Error::Internal(e.to_string()))?;

            for row in &counter_rows {
                let k: Vec<u8> = row.get("key");
                let v: i64 = row.get("value");
                if !result.iter().any(|(rk, _)| rk == &k) {
                    result.push((k, v.to_le_bytes().to_vec()));
                }
            }

            Ok(result)
        })
    }

    fn delete(&self, key: &[u8]) -> BoxFuture<'_, Result<(), Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            sqlx::query("DELETE FROM kv WHERE key = ?")
                .bind(&key)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            sqlx::query("DELETE FROM counters WHERE key = ?")
                .bind(&key)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(())
        })
    }
}

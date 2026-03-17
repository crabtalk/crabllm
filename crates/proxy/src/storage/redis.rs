use ::redis::AsyncCommands;
use crabtalk_core::{BoxFuture, Error, KvPairs, Prefix, Storage};

pub struct RedisStorage {
    conn: ::redis::aio::MultiplexedConnection,
}

impl RedisStorage {
    pub async fn open(url: &str) -> Result<Self, Error> {
        let client =
            ::redis::Client::open(url).map_err(|e| Error::Internal(format!("redis open: {e}")))?;
        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| Error::Internal(format!("redis connect: {e}")))?;
        Ok(Self { conn })
    }
}

impl Storage for RedisStorage {
    fn get(&self, key: &[u8]) -> BoxFuture<'_, Result<Option<Vec<u8>>, Error>> {
        let mut conn = self.conn.clone();
        let key = key.to_vec();
        Box::pin(async move {
            let val: Option<Vec<u8>> = conn
                .get(&key)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(val)
        })
    }

    fn set(&self, key: &[u8], value: Vec<u8>) -> BoxFuture<'_, Result<(), Error>> {
        let mut conn = self.conn.clone();
        let key = key.to_vec();
        Box::pin(async move {
            let () = conn
                .set(&key, &value)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(())
        })
    }

    fn increment(&self, key: &[u8], delta: i64) -> BoxFuture<'_, Result<i64, Error>> {
        let mut conn = self.conn.clone();
        let key = key.to_vec();
        Box::pin(async move {
            let val: i64 = conn
                .incr(&key, delta)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(val)
        })
    }

    fn list(&self, prefix: &Prefix) -> BoxFuture<'_, Result<KvPairs, Error>> {
        let mut conn = self.conn.clone();
        let pattern = {
            let mut p = prefix.to_vec();
            p.push(b'*');
            p
        };
        Box::pin(async move {
            let keys: Vec<Vec<u8>> = {
                let mut iter: ::redis::AsyncIter<Vec<u8>> = conn
                    .scan_match(&pattern)
                    .await
                    .map_err(|e| Error::Internal(e.to_string()))?;
                let mut keys = Vec::new();
                while let Some(key) = iter.next_item().await {
                    keys.push(key);
                }
                keys
            };

            if keys.is_empty() {
                return Ok(Vec::new());
            }

            let values: Vec<Option<Vec<u8>>> = ::redis::cmd("MGET")
                .arg(&keys)
                .query_async(&mut conn)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;

            let mut result = Vec::with_capacity(keys.len());
            for (key, val) in keys.into_iter().zip(values) {
                if let Some(v) = val {
                    result.push((key, v));
                }
            }

            Ok(result)
        })
    }

    fn delete(&self, key: &[u8]) -> BoxFuture<'_, Result<(), Error>> {
        let mut conn = self.conn.clone();
        let key = key.to_vec();
        Box::pin(async move {
            let () = conn
                .del(&key)
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(())
        })
    }
}

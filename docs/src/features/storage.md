# Storage

Extensions that persist data (cache, rate limits, usage, budget) use a shared
storage backend. Three backends are available.

## Memory (default)

In-memory storage using concurrent hash maps. Fast, but data is lost on restart.

```toml
[storage]
kind = "memory"
```

This is the default when no `[storage]` section is present. No feature flag
required.

## SQLite

Persistent storage using SQLite via async pooled connections.

```toml
[storage]
kind = "sqlite"
path = "crabtalk.db"
```

Requires the `storage-sqlite` feature:

```bash
cargo install crabtalk --features storage-sqlite
```

The database file is created automatically if it doesn't exist. Uses two tables
(`kv` and `counters`) with atomic increment via `INSERT ... ON CONFLICT ...
RETURNING`.

## Redis

Remote persistent storage using Redis async multiplexed connections.

```toml
[storage]
kind = "redis"
path = "redis://127.0.0.1:6379"
```

Requires the `storage-redis` feature:

```bash
cargo install crabtalk --features storage-redis
```

Supports standard Redis URLs. Increment maps to `INCRBY`, key listing uses
`SCAN` with prefix glob patterns.

## How Extensions Use Storage

Each extension namespaces its keys with a 4-byte prefix to avoid collisions:

| Extension | Operations |
|-----------|-----------|
| Cache | get/set response JSON with TTL check |
| Rate Limit | increment per-key-per-minute counters |
| Usage | increment per-key-per-model token counters |
| Budget | increment per-key spend in microdollars |

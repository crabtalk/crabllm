# Extensions

Extensions add functionality to the request pipeline via hooks. They run
in-handler (not as middleware), giving direct access to typed request and
response data.

## Available Extensions

### Cache

Caches non-streaming chat completion responses. Cache key is a SHA-256 hash of
the serialized request body.

```toml
[extensions.cache]
ttl_seconds = 3600           # default: 300 (5 minutes)
```

Admin route: `DELETE /v1/cache` — clears all cached entries.

### Rate Limit

Enforces per-key request and token rate limits using a per-minute sliding window.

```toml
[extensions.rate_limit]
requests_per_minute = 60      # required
tokens_per_minute = 100000    # optional
```

Returns HTTP 429 when limits are exceeded. Token counting uses actual usage
from provider responses (both streaming and non-streaming).

### Usage Tracker

Accumulates prompt and completion token counts per key and model.

```toml
[extensions.usage]
```

No configuration needed. Admin route: `GET /v1/usage` — returns JSON array of
usage entries with `key`, `model`, `prompt_tokens`, and `completion_tokens`.

### Budget

Enforces per-key spend limits. Requires [pricing](../configuration.md#pricing)
to be configured for the models in use.

```toml
[extensions.budget]
default_budget = 10.00        # USD, required

[extensions.budget.keys.team-a]
budget = 50.00                # USD override for this key
```

Returns HTTP 429 when a key's spend exceeds its budget. Admin route:
`GET /v1/budget` — returns JSON array with `key`, `spent_usd`, `budget_usd`,
and `remaining_usd`.

### Logging

Structured request logging via the `tracing` framework.

```toml
[extensions.logging]
level = "info"
```

Logs completed requests (model, provider, key, latency, token counts) and
errors. Initializes the `tracing_subscriber` when enabled.

## Hook Pipeline

Extensions run in config order at these points:

1. **on_request** — before provider dispatch. Can short-circuit (rate limit, budget).
2. **on_cache_lookup** — before provider dispatch for non-streaming. Returns
   cached response if available.
3. **on_response** — after successful non-streaming response.
4. **on_chunk** — for each SSE chunk during streaming.
5. **on_error** — when a provider call fails.

## Combining Extensions

Multiple extensions can be enabled simultaneously:

```toml
[extensions.logging]
level = "info"

[extensions.rate_limit]
requests_per_minute = 100

[extensions.usage]

[extensions.cache]
ttl_seconds = 600

[extensions.budget]
default_budget = 100.00
```

All extensions share the same [storage backend](storage.md).

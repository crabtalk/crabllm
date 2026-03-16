# Routing

Crabtalk decides which provider handles a request based on model name, routing
weights, and fallback logic.

## Model Resolution

When a request arrives, crabtalk looks up the model name in the configured
providers. If the model is an alias, it resolves to the canonical name first
(single-hop lookup).

## Weighted Selection

When multiple providers serve the same model, one is selected via weighted random
selection. Higher `weight` values mean more traffic:

```toml
[providers.primary]
kind = "openai_compat"
api_key = "${OPENAI_KEY_1}"
models = ["gpt-4o"]
weight = 3                    # 75% of traffic

[providers.secondary]
kind = "openai_compat"
api_key = "${OPENAI_KEY_2}"
models = ["gpt-4o"]
weight = 1                    # 25% of traffic
```

Selection is stateless — no shared counters. Each request picks independently.

## Retry

When a provider returns a transient error (HTTP 429, 500, 502, 503, 504),
crabtalk retries the same provider with exponential backoff:

- Base delay: 100ms, doubling each retry.
- Full jitter: each sleep is a random duration in `[backoff/2, backoff]` to
  prevent thundering herd.
- Max retries: configurable per provider via `max_retries` (default 2).

```toml
[providers.openai]
kind = "openai_compat"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]
max_retries = 3               # retry up to 3 times
```

Set `max_retries = 0` to disable retry entirely.

## Fallback

When retries are exhausted on a provider, crabtalk tries the next provider by
descending weight. This continues until a provider succeeds or all providers
have been tried.

```toml
# Primary provider (tried first)
[providers.openai]
kind = "openai_compat"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]
weight = 2

# Fallback provider (tried if primary fails)
[providers.azure]
kind = "azure"
api_key = "${AZURE_KEY}"
base_url = "https://my-resource.openai.azure.com"
api_version = "2024-02-01"
models = ["gpt-4o"]
weight = 1
```

## Timeouts

Each provider call is wrapped in a timeout. If the timeout expires, the request
is treated as a transient error (triggers retry/fallback):

```toml
[providers.openai]
kind = "openai_compat"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]
timeout = 60                  # seconds (default: 30)
```

Timeout errors return HTTP 504 Gateway Timeout if all providers time out.

## Streaming Behavior

For streaming requests, retry and fallback only apply to connection errors
(before the stream starts). Once the first SSE chunk is sent to the client,
the connection is committed to that provider.

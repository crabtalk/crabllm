# Authentication

Crabllm supports virtual API keys for client authentication and model access
control.

## Virtual Keys

Define keys in the config:

```toml
[[keys]]
name = "team-frontend"
key = "sk-frontend-abc123"
models = ["gpt-4o-mini"]

[[keys]]
name = "team-backend"
key = "sk-backend-xyz789"
models = ["gpt-4o", "claude-sonnet-4-20250514"]

[[keys]]
name = "admin"
key = "${ADMIN_API_KEY}"
models = ["*"]
```

Clients send the key in the `Authorization` header:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-frontend-abc123" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Model Access Control

The `models` field controls which models a key can access:

- `["gpt-4o", "gpt-4o-mini"]` — only these models.
- `["*"]` — all models.

Requests for unauthorized models return HTTP 401.

## No Auth Mode

When no keys are configured, authentication is disabled entirely. All requests
pass through without checking the `Authorization` header.

```toml
# No [[keys]] section = auth disabled
listen = "0.0.0.0:8080"

[providers.openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]
```

## Key Name Tracking

The key `name` field is used by extensions for per-key tracking:

- **Rate limiting** — enforced per key name.
- **Usage tracking** — tokens counted per key name.
- **Budget** — spend limits per key name.
- **Logging** — key name included in log entries.

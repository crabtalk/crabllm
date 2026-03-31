# Configuration

Crabllm is configured via a TOML file, passed with `--config`:

```bash
crabllm --config crabllm.toml
```

The `--bind` flag overrides the `listen` address.

## Environment Variables

Strings containing `${VAR}` are expanded from environment variables at startup.
Unknown variables expand to empty string. Use this for secrets:

```toml
api_key = "${OPENAI_API_KEY}"
```

## Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `listen` | string | required | Address to bind, e.g. `"0.0.0.0:8080"` |
| `shutdown_timeout` | integer | `30` | Graceful shutdown timeout in seconds |

## Providers

Each provider is a named entry under `[providers]`:

```toml
[providers.my_openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o", "gpt-4o-mini"]
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kind` | string | required | Provider type (see [Providers](providers/overview.md)) |
| `api_key` | string | `""` | API key for authentication |
| `base_url` | string | per-kind | Base URL override |
| `models` | list | `[]` | Model names this provider serves |
| `weight` | integer | `1` | Routing weight for load balancing |
| `max_retries` | integer | `2` | Max retries on transient errors |
| `timeout` | integer | `30` | Per-request timeout in seconds |
| `api_version` | string | — | API version (Azure only) |
| `region` | string | — | AWS region (Bedrock only) |
| `access_key` | string | — | AWS access key (Bedrock only) |
| `secret_key` | string | — | AWS secret key (Bedrock only) |

## Virtual Keys

```toml
[[keys]]
name = "team-a"
key = "sk-team-a-secret"
models = ["gpt-4o", "claude-sonnet-4-20250514"]

[[keys]]
name = "admin"
key = "sk-admin-secret"
models = ["*"]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable key name (used in usage tracking) |
| `key` | string | The bearer token clients send |
| `models` | list | Allowed models. `["*"]` means all |

When no keys are configured, authentication is disabled.

## Aliases

```toml
[aliases]
gpt4 = "gpt-4o"
claude = "claude-sonnet-4-20250514"
```

Maps friendly model names to canonical names. Single-hop lookup.

## Pricing

```toml
[pricing.gpt-4o]
prompt_cost_per_million = 2.50
completion_cost_per_million = 10.00

[pricing.claude-sonnet-4-20250514]
prompt_cost_per_million = 3.00
completion_cost_per_million = 15.00
```

Per-model token pricing in USD. Used by the budget extension for spend tracking.

## Extensions

```toml
[extensions.cache]
ttl = 3600

[extensions.rate_limit]
rpm = 60

[extensions.usage]

[extensions.budget]
default_limit = 10000000

[extensions.logging]
level = "info"
```

See [Extensions](features/extensions.md) for details on each.

## Storage

```toml
[storage]
kind = "memory"
```

| Kind | Feature flag | `path` field |
|------|-------------|--------------|
| `memory` | none (default) | not used |
| `sqlite` | `storage-sqlite` | file path, e.g. `"crabllm.db"` |
| `redis` | `storage-redis` | URL, e.g. `"redis://127.0.0.1:6379"` |

See [Storage](features/storage.md) for details.

## Full Example

```toml
listen = "0.0.0.0:8080"
shutdown_timeout = 30

[providers.openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o", "gpt-4o-mini"]
weight = 2
max_retries = 2
timeout = 30

[providers.anthropic]
kind = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-4-20250514"]

[providers.ollama]
kind = "ollama"
models = ["llama3.2"]

[aliases]
gpt4 = "gpt-4o"
claude = "claude-sonnet-4-20250514"

[[keys]]
name = "default"
key = "${CRABTALK_API_KEY}"
models = ["*"]

[pricing.gpt-4o]
prompt_cost_per_million = 2.50
completion_cost_per_million = 10.00

[extensions.rate_limit]
rpm = 100

[extensions.usage]

[extensions.logging]
level = "info"

[storage]
kind = "sqlite"
path = "crabllm.db"
```

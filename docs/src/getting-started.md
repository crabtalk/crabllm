# Getting Started

## Install

```bash
cargo install crabllm
```

## Configure

Create a `crabllm.toml` file:

```toml
listen = "0.0.0.0:8080"

[providers.openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o", "gpt-4o-mini"]

[providers.anthropic]
kind = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-4-20250514"]
```

Environment variables in `${VAR}` syntax are expanded at startup.

## Run

```bash
crabllm --config crabllm.toml
```

You'll see:

```
crabllm listening on 0.0.0.0:8080 (3 models, 2 providers, 0 extensions)
```

## Send a Request

All requests use the OpenAI format, regardless of which provider handles them:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

To use Anthropic, just change the model name:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The request format is the same. Crabllm translates it to the Anthropic Messages
API internally.

## Streaming

Add `"stream": true` to get SSE streaming:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Model Aliasing

Map friendly names to canonical model names:

```toml
[aliases]
gpt4 = "gpt-4o"
claude = "claude-sonnet-4-20250514"
```

Now `"model": "gpt4"` routes to `gpt-4o`.

## Next Steps

- [Configuration](configuration.md) — full reference for all config options
- [Providers](providers/overview.md) — setup guides for each provider
- [Features](features/routing.md) — routing, auth, extensions, and more

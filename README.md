# crabtalk

High-performance LLM API gateway in Rust. Inspired by
[LiteLLM](https://github.com/BerriAI/litellm).

## What It Does

Crabtalk sits between your application and LLM providers. It exposes an
OpenAI-compatible API and routes requests to the configured provider —
OpenAI, Anthropic, Azure, Ollama, and any OpenAI-compatible service.

One API format. Many providers. Low overhead.

## Features

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- Multi-provider support (OpenAI, Anthropic, Azure, Ollama, vLLM, Groq, and more)
- SSE streaming pass-through
- Anthropic Messages API translation (automatic OpenAI ↔ Anthropic format conversion)
- API key authentication with virtual keys
- TOML configuration with `${ENV_VAR}` interpolation
- Single static binary

## Quick Start

```bash
cargo install crabtalk
```

Create `crabtalk.toml`:

```toml
listen = "0.0.0.0:8080"

[providers.openai]
kind = "openai_compat"
api_key = "${OPENAI_API_KEY}"

[providers.anthropic]
kind = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"

[models.gpt-4o]
provider = "openai"

[models.claude-sonnet-4-20250514]
provider = "anthropic"
```

Run:

```bash
crabtalk --config crabtalk.toml
```

Send requests using the OpenAI format:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Why Rust

- **Performance**: Sub-millisecond proxy overhead. No GC pauses.
- **Safety**: Memory safety without runtime cost.
- **Concurrency**: Tokio async runtime handles thousands of concurrent
  streaming connections.
- **Deployment**: Single static binary. No interpreter, no virtualenv.

## Crates

| Crate | Description |
|-------|-------------|
| `crabtalk` | Binary entry point (CLI + server startup) |
| `crabtalk-core` | Shared types: config, OpenAI-format request/response, errors |
| `crabtalk-provider` | Provider enum, registry, and upstream HTTP dispatch |
| `crabtalk-proxy` | Axum HTTP server, route handlers, auth middleware |

## License

MIT OR Apache-2.0

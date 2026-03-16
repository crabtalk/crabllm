# crabtalk

[![crates.io][crabtalk-badge]][crabtalk-crate]

High-performance LLM API gateway in Rust, used by [OpenWalrus][openwalrus].

## What It Does

Crabtalk sits between your application and LLM providers. It exposes an
OpenAI-compatible API and routes requests to the configured provider —
OpenAI, Anthropic, Azure, Ollama, and any OpenAI-compatible service.

One API format. Many providers. Low overhead.

## Features

Inspired by [LiteLLM][litellm]. Built in Rust for minimal overhead.

| Feature | LiteLLM | Crabtalk |
|---------|:-------:|:--------:|
| `/chat/completions` | ✅ | ✅ |
| `/embeddings` | ✅ | ✅ |
| `/models` | ✅ | ✅ |
| OpenAI provider | ✅ | ✅ |
| Anthropic provider | ✅ | ✅ |
| Google Gemini provider | ✅ | ✅ |
| Azure OpenAI provider | ✅ | ✅ |
| AWS Bedrock provider | ✅ | ✅ |
| Tool/function calling | ✅ | ✅ |
| SSE streaming | ✅ | ✅ |
| Virtual keys + auth | ✅ | ✅ |
| Weighted routing | ✅ | ✅ |
| Model aliasing | ✅ | ✅ |
| Retry + fallback | ✅ | ✅ |
| Request timeouts | ✅ | ✅ |
| Rate limiting (RPM) | ✅ | ✅ |
| Cost/usage tracking | ✅ | ✅ |
| Budget enforcement | ✅ | ✅ |
| Request caching | ✅ | ✅ |
| Graceful shutdown | ✅ | ✅ |
| Storage (memory) | ✅ | ✅ |
| Storage (persistent) | ✅ Postgres | ✅ SQLite |
| `/completions` (text) | ✅ | — |
| Image/audio endpoints | ✅ | — |
| Rate limiting (TPM) | ✅ | ✅ |
| Redis storage | ✅ | ✅ |
| Dashboard/UI | ✅ | — |

[litellm]: https://github.com/BerriAI/litellm

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
models = ["gpt-4o"]

[providers.anthropic]
kind = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-4-20250514"]
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

| Crate               | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `crabtalk`          | Binary entry point (CLI + server startup)                    |
| `crabtalk-core`     | Shared types: config, OpenAI-format request/response, errors |
| `crabtalk-provider` | Provider enum, registry, and upstream HTTP dispatch          |
| `crabtalk-proxy`    | Axum HTTP server, route handlers, auth middleware            |

## License

MIT OR Apache-2.0

[crabtalk-badge]: https://img.shields.io/crates/v/crabtalk.svg
[crabtalk-crate]: https://crates.io/crates/crabtalk
[openwalrus]: https://openwalrus.xyz

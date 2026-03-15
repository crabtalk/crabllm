# crabtalk

High-performance LLM API gateway in Rust. Inspired by
[LiteLLM](https://github.com/BerriAI/litellm).

## What It Does

Crabtalk sits between your application and LLM providers. It exposes an
OpenAI-compatible API and routes requests to the configured provider —
OpenAI, Anthropic, Google, AWS Bedrock, Azure, Ollama, and others.

One API format. Many providers. Low overhead.

## Why Rust

- **Performance**: Sub-millisecond proxy overhead. No GC pauses.
- **Safety**: Memory safety without runtime cost.
- **Concurrency**: Tokio async runtime handles thousands of concurrent
  streaming connections.
- **Deployment**: Single static binary. No interpreter, no virtualenv.

## Planned Features

- OpenAI-compatible API (`/chat/completions`, `/completions`, `/embeddings`)
- Multi-provider routing (OpenAI, Anthropic, Google, AWS Bedrock, Azure, Ollama)
- SSE streaming pass-through
- Load balancing and failover
- API key management and virtual keys
- Cost tracking per key/team
- Configuration-driven setup (TOML)

## Status

Early development. Not usable yet.

## License

MIT OR Apache-2.0

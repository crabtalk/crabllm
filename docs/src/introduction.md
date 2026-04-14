# Introduction

Crabllm is a high-performance LLM API gateway written in Rust. It sits between
your application and LLM providers, exposing an OpenAI-compatible API surface.

One API format. Many providers. Low overhead.

## What It Does

You send requests in OpenAI format to crabllm. It routes them to the configured
provider — OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, or
Ollama — translating the request and response as needed.

The full HTTP API surface is documented interactively at [crabtalk.github.io/crabllm/api](https://crabtalk.github.io/crabllm/api).

Your application talks to one endpoint. Crabllm handles the rest:

- **Provider translation** — Anthropic, Google, and Bedrock have their own API
  formats. Crabllm translates automatically.
- **Routing** — Weighted random selection across multiple providers for the same
  model. Automatic fallback when a provider fails.
- **Streaming** — SSE streaming proxied without buffering.
- **Auth** — Virtual API keys with per-key model access control.
- **Extensions** — Rate limiting, caching, cost tracking, budget enforcement.

## Why Rust

- **Sub-millisecond overhead** — no GC pauses, no interpreter startup.
- **Memory safety** — without runtime cost.
- **Concurrency** — Tokio async runtime handles thousands of concurrent
  streaming connections efficiently.
- **Deployment** — single static binary. No interpreter, no virtualenv, no
  Docker required.

## Feature Comparison

| Feature | LiteLLM | Crabllm |
|---------|:-------:|:--------:|
| `/chat/completions` | yes | yes |
| `/embeddings` | yes | yes |
| `/models` | yes | yes |
| OpenAI provider | yes | yes |
| Anthropic provider | yes | yes |
| Google Gemini provider | yes | yes |
| Azure OpenAI provider | yes | yes |
| AWS Bedrock provider | yes | yes |
| Tool/function calling | yes | yes |
| SSE streaming | yes | yes |
| Virtual keys + auth | yes | yes |
| Weighted routing | yes | yes |
| Model aliasing | yes | yes |
| Retry + fallback | yes | yes |
| Rate limiting (RPM/TPM) | yes | yes |
| Cost/usage tracking | yes | yes |
| Budget enforcement | yes | yes |
| Request caching | yes | yes |
| Image/audio endpoints | yes | yes |
| Storage (memory) | yes | yes |
| Storage (persistent) | Postgres | SQLite |
| Redis storage | yes | yes |

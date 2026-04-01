# Contributing

Crabllm is an LLM API gateway. It takes OpenAI-format requests and translates
them to provider-native formats. The gateway adds no business logic — it
translates, routes, and streams.

## Layering

```
Layer 0 ─ Core
  └─ core             Types, config, error — no I/O, no provider logic

Layer 1 ─ Providers (independent of each other)
  ├─ provider         Dispatch, request/response translation per provider
  └─ llamacpp         Managed llama.cpp server process

Layer 2 ─ Gateway
  └─ proxy            HTTP server (axum), routing, streaming, extensions

Layer 3 ─ Binary
  └─ crabllm          Config loading, server startup
```

## Where does my change go?

| Question | Crate |
|----------|-------|
| Does it define a shared type, config struct, or error? | core |
| Does it add or fix a provider's translation layer? | provider |
| Does it change HTTP handling, routing, or middleware? | proxy |
| Does it touch llama.cpp process management? | llamacpp |
| **If none of these fit, challenge whether the change should exist.** | |

## Boundary Contracts

- **Core** — types and config only. No network I/O, no provider-specific logic.
  If a core change pulls in provider or proxy deps, the abstraction is wrong.
- **Provider** — translates requests and responses. Never touches HTTP routing
  or gateway concerns. Each provider is self-contained.
- **Proxy** — routes and streams. Never interprets provider-specific formats.

## Data Flow

```
Client → POST /chat/completions (OpenAI format)
  → Proxy: auth, routing, model resolution
  → Provider: translate request → provider API → translate response
  → Proxy: stream SSE back to client
```

## Pull Requests

- One logical change per PR. Don't mix features, refactors, and dependency bumps.
- Break work into reviewable commits — each commit should have a single reason
  to exist.
- PR titles use conventional commits: `type(scope): description`.
- Don't vendor dependencies. Fix upstream instead.

## Design

See the [docs](https://crabtalk.github.io/crabllm/)
([source](docs/src/SUMMARY.md)) for provider details, routing, extensions,
and configuration.

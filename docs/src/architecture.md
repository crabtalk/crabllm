# Architecture

## Principles

- **Simplicity over abstraction.** No trait where a function suffices.
- **Single responsibility.** Each crate has one focused job.
- **OpenAI as canonical format.** Providers translate to/from it.
- **Streaming first-class.** Never buffer a full response when streaming.
- **Configuration-driven.** Provider setup and routing from config, not code.
- **Minimal gateway latency.** Avoid hot-path allocations.

## Workspace Layout

```
crabtalk/
  crates/
    crabtalk/   — binary, wires everything together
    core/       — shared types, config, errors
    provider/   — provider enum + translation modules
    proxy/      — HTTP server, routing, extensions
    bench/      — benchmark mock backend
```

## Crates

### crabtalk

Binary entry point. Loads TOML config, builds the provider registry, initializes
the storage backend and extensions, starts the Axum HTTP server. CLI args:
`--config` and `--bind`.

### core

Shared types with no business logic. Contains:
- **Config** — `GatewayConfig` with env var interpolation.
- **Types** — OpenAI-compatible wire format structs (request, response, chunk).
- **Error** — error enum with transient detection for retry logic.
- **Storage** — async KV trait with memory, SQLite, and Redis backends.
- **Extension** — hook trait for the request pipeline.

### provider

Provider dispatch. The `Provider` enum has variants for each supported provider.
Each variant dispatches to a per-provider module that handles request/response
translation. `ProviderRegistry` maps model names to weighted deployment lists.

### proxy

Axum HTTP server. Route handlers implement retry + fallback across deployments.
Auth middleware validates virtual keys. Five built-in extensions run as
in-handler hooks.

## Request Flow

1. Client sends OpenAI-format request to crabtalk.
2. Auth middleware validates the bearer token.
3. Handler resolves model name (aliases) and gets deployment list.
4. Extension `on_request` hooks run (rate limit, budget check).
5. Cache lookup for non-streaming requests.
6. Provider dispatch with retry + fallback.
7. Provider translates request, calls upstream, translates response.
8. Extension `on_response`/`on_chunk` hooks run (usage, budget, cache store).
9. Response returned to client.

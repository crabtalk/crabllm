# crabllm

[![crates.io][crabllm-badge]][crabllm-crate]

High-performance LLM API gateway in Rust, used by [OpenWalrus][openwalrus].

## What It Does

Crabllm sits between your application and LLM providers. It exposes an
OpenAI-compatible API and routes requests to the configured provider —
OpenAI, Anthropic, Azure, Ollama, and any OpenAI-compatible service.

One API format. Many providers. Low overhead.

Inspired by [LiteLLM][litellm]. Built in Rust for minimal overhead.
See the [docs][docs] for [providers][providers], [routing][routing],
[extensions][extensions], and [configuration][configuration].


## Quick Start

```bash
cargo install crabllm
```

Create `crabllm.toml`:

```toml
listen = "0.0.0.0:8080"

[providers.openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]

[providers.anthropic]
kind = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-4-20250514"]
```

Run:

```bash
crabllm --config crabllm.toml
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

## Benchmarks

Gateway overhead measured against a mock LLM server with instant responses.
See [full results][benchmarks] for streaming, embeddings, and memory.

**Chat completions P50 (ms) — lower is better:**

| RPS | direct | crabllm | [bifrost] | [litellm] |
|----:|-------:|--------:|----------:|----------:|
| 500 | 0.28 | 0.66 | 0.36 | 168.79 |
| 1000 | 0.15 | 0.44 | 0.27 | 172.00 |
| 5000 | 0.13 | 0.26 | 0.26 | 159.86 |

```bash
# requires linux + docker
make bench                         # full competitive benchmark
make bench-debug                   # quick bifrost-only debug run
make bench-chart                   # render terminal charts from results
make bench-json                    # dump summary JSON to stdout
make summary                       # generate docs/src/benchmarks.md
```

[bifrost]: https://github.com/maximhq/bifrost
[benchmarks]: https://clearloop.github.io/crabllm/benchmarks.html

## Crates

| Crate               | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `crabllm`          | Binary entry point (CLI + server startup)                    |
| `crabllm-core`     | Shared types: config, OpenAI-format request/response, errors |
| `crabllm-provider` | Provider enum, registry, and upstream HTTP dispatch          |
| `crabllm-proxy`    | Axum HTTP server, route handlers, auth middleware            |

## License

MIT OR Apache-2.0


[crabllm-badge]: https://img.shields.io/crates/v/crabllm.svg
[crabllm-crate]: https://crates.io/crates/crabllm
[openwalrus]: https://openwalrus.xyz
[litellm]: https://github.com/BerriAI/litellm
[docs]: https://clearloop.github.io/crabllm
[providers]: https://clearloop.github.io/crabllm/providers/overview.html
[routing]: https://clearloop.github.io/crabllm/features/routing.html
[extensions]: https://clearloop.github.io/crabllm/features/extensions.html
[configuration]: https://clearloop.github.io/crabllm/configuration.html
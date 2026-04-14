# crabllm

[![crates.io][crabllm-badge]][crabllm-crate]

High-performance LLM API gateway in Rust, built by [crabtalk][crabtalk].

## What It Does

Crabllm sits between your application and LLM providers. It exposes an
OpenAI-compatible API and routes requests to the configured provider —
OpenAI, Anthropic, Azure, Ollama, and any OpenAI-compatible service.

One API format. Many providers. Low overhead.

See the [docs][docs] for [providers][providers], [routing][routing],
[extensions][extensions], and [configuration][configuration].


## Quick Start

```bash
cargo install crabllm crabctl
crabllm serve
```

First run generates `crabllm.toml` with a fresh admin token and default
key. Add a provider — no restart needed:

```bash
crabctl providers create openai --kind openai --api-key "$OPENAI_API_KEY"
```

With `--models` omitted the server auto-populates the list from the
provider's `/models` endpoint. Send a request using the OpenAI format:

```bash
curl http://127.0.0.1:5632/v1/chat/completions \
  -H "Authorization: Bearer $(grep -m1 -oE 'sk-[a-z0-9]+' crabllm.toml | tail -1)" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'
```

Prefer Docker? See the [Docker chapter][docker] — pull
`ghcr.io/crabtalk/crabllm:latest`, mount a volume, and manage providers
the same way with `crabctl`.

## Benchmarks

Gateway overhead measured against a mock LLM server with instant responses.
Numbers show proxy cost only (gateway latency minus direct baseline).
See [full results][benchmarks] for all scenarios and memory usage.

**Streaming P50 overhead (ms) — the metric that matters for LLMs:**

| RPS | crabllm | [bifrost] | [litellm] |
|----:|--------:|----------:|----------:|
| 500 | +0.02 | +0.16 | +593.20 |
| 1000 | +0.09 | +0.18 | +596.90 |
| 5000 | +0.23 | +0.47 | +593.85 |

**Chat completions P50 overhead (ms):**

| RPS | crabllm | [bifrost] | [litellm] |
|----:|--------:|----------:|----------:|
| 500 | +0.41 | +0.07 | +151.42 |
| 1000 | +0.30 | +0.10 | +160.12 |
| 5000 | +0.16 | +0.14 | +159.68 |

**Peak memory:** crabllm 37MB · bifrost 169MB · litellm 541MB

```bash
# requires linux + docker
make bench                         # full competitive benchmark
make bench-debug GW=crabllm        # quick single-gateway debug run
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
[crabtalk]: https://crabtalk.xyz
[litellm]: https://github.com/BerriAI/litellm
[docs]: https://clearloop.github.io/crabllm
[docker]: https://clearloop.github.io/crabllm/docker.html
[providers]: https://clearloop.github.io/crabllm/providers/overview.html
[routing]: https://clearloop.github.io/crabllm/features/routing.html
[extensions]: https://clearloop.github.io/crabllm/features/extensions.html
[configuration]: https://clearloop.github.io/crabllm/configuration.html
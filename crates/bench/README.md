# crabllm-bench

Mock OpenAI-compatible backend and competitive benchmark suite for crabllm.
Compares gateway overhead against LiteLLM and Bifrost using canned responses
with zero LLM latency.

## Quick start

Requires Linux (host networking) and pre-built prod binaries.

```sh
# Build prod binaries
cargo build --profile prod -p crabllm -p crabllm-bench

# Run the full benchmark
make bench

# Run a specific group with custom duration
make bench ARGS="--group overhead --duration 10 --rps '100 500 1000'"

# View charts from existing results
make bench-chart

# Tear down containers
cd crates/bench && docker compose down
```

## Benchmark groups

| Group | Scenarios | What it measures |
|---|---|---|
| `overhead` | chat-minimal, chat-stream, embeddings | Pure gateway latency at 100–5000 RPS |
| `payload` | large-context (50 msgs), long-stream, tools (10 defs) | Serialization overhead on big payloads |
| `concurrent` | 100/500/1000 concurrent streams | Connection handling under load |

## Architecture

All services run on host networking for zero Docker NAT overhead.

| Service | Port | Image |
|---|---|---|
| mock | 9999 | crabllm-bench (canned OpenAI responses) |
| crabllm | 6666 | crabllm gateway |
| bifrost | 6668 | maximhq/bifrost (Go) |
| litellm | 4000 | ghcr.io/berriai/litellm (Python) |
| runner | — | oha load generator + bench.py orchestrator |

Each gateway gets identical resource limits (2 CPUs, 512MB).

## Results

Results are saved to `crates/bench/results/` as JSON files from
[oha](https://github.com/hatoo/oha). Memory usage is tracked in
`.mem.json` sidecar files.

```sh
# Terminal charts
python3 bench.py --chart-only --output results

# PNG export (requires matplotlib)
python3 bench.py --chart-only --png --output results
```

## Mock backend

The mock serves canned responses with configurable streaming behavior:

```sh
crabllm-bench --port 9999 --chunks 20 --delay 0 --error-rate 0
```

- `--chunks` — SSE events per streaming response
- `--delay` — milliseconds between chunks (or before non-streaming response)
- `--error-rate` — fraction of requests that return 500 (0.0–1.0)

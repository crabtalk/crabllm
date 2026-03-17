# crabtalk-bench

Mock OpenAI-compatible backend for benchmarking crabtalk. Serves canned
responses with zero LLM latency so you can measure gateway overhead in
isolation.

## Endpoints

- `POST /v1/chat/completions` — non-streaming JSON or SSE streaming
- `POST /v1/embeddings` — 1536-dim zero vector
- `GET /v1/models` — two mock models (`bench-chat`, `bench-embed`)

## Usage

```sh
cargo install crabtalk-bench
crabtalk-bench --port 9999 --chunks 20
```

`--chunks` controls how many SSE events a streaming response produces.

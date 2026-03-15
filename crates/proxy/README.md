# crabtalk-proxy

HTTP proxy server for the [crabtalk](https://github.com/clearloop/crabtalk) LLM API gateway.

Axum-based HTTP server providing:

- `POST /v1/chat/completions` — Non-streaming and SSE streaming
- `POST /v1/embeddings` — Embedding requests
- `GET /v1/models` — List configured models
- Bearer token authentication middleware

## License

MIT OR Apache-2.0

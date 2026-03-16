# crabtalk-proxy

[![crates.io][badge]][crate]

HTTP proxy server for the [crabtalk](https://github.com/clearloop/crabtalk) LLM API gateway.

Axum-based HTTP server providing:

- `POST /v1/chat/completions` — non-streaming and SSE streaming with retry + fallback
- `POST /v1/embeddings` — embedding requests
- `POST /v1/images/generations` — image generation
- `POST /v1/audio/speech` — text-to-speech
- `POST /v1/audio/transcriptions` — audio transcription
- `GET /v1/models` — list configured models
- Bearer token authentication middleware
- Extension system: cache, rate limiting, usage tracking, budget, logging

## License

MIT OR Apache-2.0

[badge]: https://img.shields.io/crates/v/crabtalk-proxy.svg
[crate]: https://crates.io/crates/crabtalk-proxy

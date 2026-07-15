# Kimi

The `kimi` provider routes to [Moonshot AI](https://www.moonshot.ai)'s Kimi
models. It exposes an OpenAI-compatible endpoint for chat, streaming, and
embeddings, and a native Anthropic-compatible endpoint for Anthropic-format
passthrough. Authentication is a bearer token on both.

## Configuration

```toml
[providers.kimi]       # section name doubles as kind; add `kind = "kimi"` only if you rename it
api_key = "${MOONSHOT_API_KEY}"
models = ["kimi-k2-0711-preview"]
```

Defaults, no `base_url` needed:

- OpenAI-compatible: `https://api.moonshot.ai/v1`
- Anthropic-compatible: `https://api.moonshot.ai/anthropic/v1` (the `/v1/messages`
  path the Anthropic SDK / Claude Code resolve from `ANTHROPIC_BASE_URL`)

Set `base_url` to override the OpenAI-compatible origin (e.g. the mainland-China
endpoint `https://api.moonshot.cn/v1`).

## Usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-0711-preview",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Anthropic-format requests are served natively by Moonshot's Anthropic-compatible
endpoint rather than translated.

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings
- Anthropic Messages (native passthrough, streaming and non-streaming)
- Tool/function calling

## Limitations

- Moonshot rescales the sampling temperature (`real = requested * 0.6`), so a
  given `temperature` behaves cooler than on other providers.

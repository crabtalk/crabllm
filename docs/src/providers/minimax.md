# MiniMax

The `minimax` provider routes to [MiniMax](https://www.minimax.io)'s models. It
exposes an OpenAI-compatible endpoint for chat, streaming, and embeddings, and a
native Anthropic-compatible endpoint for Anthropic-format passthrough.
Authentication is a bearer token on both.

## Configuration

```toml
[providers.minimax]    # section name doubles as kind; add `kind = "minimax"` only if you rename it
api_key = "${MINIMAX_API_KEY}"
models = ["MiniMax-M2"]
```

Defaults, no `base_url` needed:

- OpenAI-compatible: `https://api.minimax.io/v1`
- Anthropic-compatible: `https://api.minimax.io/anthropic/v1` (the `/v1/messages`
  path the Anthropic SDK / Claude Code resolve from `ANTHROPIC_BASE_URL`)

Set `base_url` to override the OpenAI-compatible origin (e.g. the `api.minimaxi.com`
region).

## Usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Anthropic-format requests are served natively by MiniMax's Anthropic-compatible
endpoint rather than translated.

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings
- Anthropic Messages (native passthrough, streaming and non-streaming)
- Tool/function calling

## Limitations

- The Anthropic-compatible endpoint reports a smaller `context_window` in model
  metadata than some MiniMax models actually support; clients that trust that
  metadata (e.g. Claude Code) may cap their budget early.

# DeepSeek

The `deepseek` provider routes to [DeepSeek](https://deepseek.com)'s models.
DeepSeek exposes two compatible surfaces and crabllm uses both: an
OpenAI-compatible endpoint for chat and streaming, and a native
Anthropic-compatible endpoint for Anthropic-format passthrough. Authentication is
a bearer token on both.

## Configuration

```toml
[providers.deepseek]   # section name doubles as kind; add `kind = "deepseek"` only if you rename it
api_key = "${DEEPSEEK_API_KEY}"
models = ["deepseek-chat", "deepseek-reasoner"]
```

Defaults, no `base_url` needed:

- OpenAI-compatible: `https://api.deepseek.com/v1`
- Anthropic-compatible: `https://api.deepseek.com/anthropic`

Set `base_url` to override the origin (both endpoints derive from it).

## Usage

Send requests in OpenAI format as usual — just set the model to a DeepSeek id:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Anthropic-format requests are served natively by DeepSeek's Anthropic-compatible
endpoint rather than translated.

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Anthropic Messages (native passthrough, streaming and non-streaming)
- Tool/function calling

## Limitations

- Embeddings, image generation, and audio endpoints are not offered by DeepSeek.

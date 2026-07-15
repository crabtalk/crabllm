# z.ai

The `zai` provider routes to [z.ai](https://z.ai)'s GLM models. z.ai exposes two
compatible surfaces and crabllm uses both: an OpenAI-compatible endpoint for chat,
streaming, and embeddings, and a native Anthropic-compatible endpoint for
Anthropic-format passthrough. Authentication is a bearer token on both.

## Configuration

```toml
[providers.zai]        # section name doubles as kind; add `kind = "zai"` only if you rename it
api_key = "${ZAI_API_KEY}"
models = ["glm-4.7", "glm-4.7-flash", "glm-5"]
```

Defaults, no `base_url` needed:

- OpenAI-compatible: `https://api.z.ai/api/paas/v4`
- Anthropic-compatible: `https://api.z.ai/api/anthropic/v1` (the `/v1/messages` path
  the Anthropic SDK / Claude Code resolve from `ANTHROPIC_BASE_URL`)

Set `base_url` to override the OpenAI-compatible origin (e.g. a regional or
proxied endpoint).

## Usage

Send requests in OpenAI format as usual — just set the model to a GLM id:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Anthropic-format requests are served natively by z.ai's Anthropic-compatible
endpoint rather than translated.

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings
- Anthropic Messages (native passthrough, streaming and non-streaming)
- Tool/function calling

## Limitations

- Image generation and audio endpoints are not offered by z.ai.

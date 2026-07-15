# Qwen

The `qwen` provider routes to Alibaba's [Qwen](https://qwen.ai) models via
DashScope. Qwen exposes two compatible surfaces and crabllm uses both: an
OpenAI-compatible endpoint for chat, streaming, and embeddings, and a native
Anthropic-compatible endpoint for Anthropic-format passthrough. Authentication is
a bearer token on both (the Anthropic endpoint also accepts `x-api-key`).

## Configuration

```toml
[providers.qwen]       # section name doubles as kind; add `kind = "qwen"` only if you rename it
api_key = "${DASHSCOPE_API_KEY}"
models = ["qwen-max", "qwen-plus", "qwen-turbo"]
```

Defaults, no `base_url` needed (international / `-intl` region):

- OpenAI-compatible: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- Anthropic-compatible: `https://dashscope-intl.aliyuncs.com/apps/anthropic/v1` (the
  `/v1/messages` path the Anthropic SDK / Claude Code resolve from `base_url`)

Set `base_url` to override the OpenAI-compatible origin — e.g. the mainland-China
endpoint `https://dashscope.aliyuncs.com/compatible-mode/v1`.

## Usage

Send requests in OpenAI format as usual — just set the model to a Qwen id:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-max",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Anthropic-format requests are served natively by Qwen's Anthropic-compatible
endpoint rather than translated.

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings
- Anthropic Messages (native passthrough, streaming and non-streaming)
- Tool/function calling

## Limitations

- The Anthropic-compatible endpoint serves only `/v1/messages` — it has no
  `/v1/models` discovery route, which some Anthropic-native tools expect.

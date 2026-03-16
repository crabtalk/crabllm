# Anthropic

The `anthropic` provider translates OpenAI-format requests to the Anthropic
Messages API and back.

## Configuration

```toml
[providers.anthropic]
kind = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-4-20250514", "claude-haiku-4-20250514"]
```

## Translation

Crabtalk handles the full translation between OpenAI and Anthropic formats:

- **System messages** — extracted from the messages array and sent as the
  Anthropic `system` parameter.
- **Stop reasons** — mapped between formats (`end_turn` to `stop`, etc.).
- **Tool calling** — fully supported. Tool definitions, tool use responses,
  and tool result messages are all translated.
- **Streaming** — Anthropic's event stream (`message_start`, `content_block_delta`,
  etc.) is translated to OpenAI-format SSE chunks.

## Usage

Send requests in OpenAI format as usual:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Limitations

- Embeddings, image generation, and audio endpoints are not supported by the
  Anthropic API.

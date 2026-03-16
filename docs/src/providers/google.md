# Google Gemini

The `google` provider translates OpenAI-format requests to the Google Gemini API
(generativeai).

## Configuration

```toml
[providers.google]
kind = "google"
api_key = "${GOOGLE_API_KEY}"
models = ["gemini-2.0-flash", "gemini-2.5-pro"]
```

## Translation

- **System messages** — mapped to Gemini's `systemInstruction` field.
- **Roles** — `assistant` mapped to `model`, `user` stays `user`.
- **Content** — mapped to Gemini's `parts` array format.
- **Tool calling** — tool definitions mapped to `functionDeclarations`, tool
  messages to `functionResponse` parts, responses extract `functionCall` parts.
- **Streaming** — uses `streamGenerateContent?alt=sse` and translates the
  Gemini event stream to OpenAI-format SSE chunks.

## Usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Limitations

- Embeddings, image generation, and audio endpoints are not supported.

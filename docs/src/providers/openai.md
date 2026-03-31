# OpenAI

The `openai` provider works with OpenAI and any OpenAI-compatible API
(Groq, Together AI, vLLM, etc.). Requests are forwarded as-is with URL and auth
rewrite — no translation needed.

## Configuration

```toml
[providers.openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o", "gpt-4o-mini", "text-embedding-3-small"]
```

## Custom Base URL

For OpenAI-compatible services, set `base_url`:

```toml
[providers.groq]
kind = "openai"
api_key = "${GROQ_API_KEY}"
base_url = "https://api.groq.com/openai/v1"
models = ["llama-3.3-70b-versatile"]

[providers.together]
kind = "openai"
api_key = "${TOGETHER_API_KEY}"
base_url = "https://api.together.xyz/v1"
models = ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"]
```

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings
- Image generation
- Audio speech (TTS)
- Audio transcription

## Tool Calling

Tool calling works as-is — the request body is forwarded directly:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

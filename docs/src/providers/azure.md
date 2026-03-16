# Azure OpenAI

The `azure` provider routes to Azure OpenAI deployments. The request body is
OpenAI-format (no translation needed), but the URL pattern and authentication
differ.

## Configuration

```toml
[providers.azure]
kind = "azure"
api_key = "${AZURE_OPENAI_KEY}"
base_url = "https://my-resource.openai.azure.com"
api_version = "2024-02-01"
models = ["gpt-4o"]
```

- `base_url` — your Azure OpenAI resource URL.
- `api_version` — the Azure API version string.

## How It Works

Crabtalk rewrites the URL to Azure's deployment-based pattern:

```
POST /openai/deployments/{model}/chat/completions?api-version={api_version}
```

Authentication uses the `api-key` header instead of `Authorization: Bearer`.

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings
- Image generation
- Audio speech (TTS)
- Audio transcription

## Usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

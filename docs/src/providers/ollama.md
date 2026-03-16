# Ollama

The `ollama` provider connects to a local [Ollama](https://ollama.ai) instance.
Ollama exposes an OpenAI-compatible API, so requests are forwarded as-is.

## Configuration

```toml
[providers.ollama]
kind = "ollama"
models = ["llama3.2", "mistral"]
```

The default base URL is `http://localhost:11434/v1`. Override it if Ollama runs
on a different host:

```toml
[providers.ollama]
kind = "ollama"
base_url = "http://192.168.1.100:11434/v1"
models = ["llama3.2"]
```

No API key is needed for local Ollama.

## Usage

Start Ollama, pull a model, then send requests through crabtalk:

```bash
ollama pull llama3.2
crabtalk --config crabtalk.toml
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Supported Endpoints

- Chat completions (streaming and non-streaming)
- Embeddings (if supported by the Ollama model)

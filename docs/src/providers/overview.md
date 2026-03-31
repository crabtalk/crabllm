# Providers

A provider is an LLM service that crabllm routes requests to. Each provider has
its own API format and authentication mechanism. Crabllm translates between the
OpenAI-compatible format your application uses and the provider's native format.

## Supported Providers

| Kind | Provider | Translation |
|------|----------|------------|
| `openai` | OpenAI, Groq, Together, vLLM, any OpenAI-compatible API | Pass-through |
| `anthropic` | Anthropic Messages API | Full translation |
| `google` | Google Gemini | Full translation |
| `azure` | Azure OpenAI | URL + auth rewrite |
| `bedrock` | AWS Bedrock Converse API | Full translation + SigV4 signing |
| `ollama` | Ollama (local models) | Pass-through (OpenAI-compatible) |

## Common Fields

Every provider supports these fields:

```toml
[providers.name]
kind = "..."           # required
api_key = "..."        # API key (supports ${ENV_VAR})
base_url = "..."       # base URL override
models = ["..."]       # model names this provider serves
weight = 1             # routing weight (higher = more traffic)
max_retries = 2        # retries on transient errors (429, 5xx)
timeout = 30           # per-request timeout in seconds
```

## Multiple Providers for the Same Model

When multiple providers list the same model, crabllm selects between them using
weighted random selection. If the selected provider fails, it falls back to the
next provider by weight. See [Routing](../features/routing.md).

```toml
[providers.openai_primary]
kind = "openai"
api_key = "${OPENAI_KEY_1}"
models = ["gpt-4o"]
weight = 3

[providers.openai_backup]
kind = "openai"
api_key = "${OPENAI_KEY_2}"
models = ["gpt-4o"]
weight = 1
```

## Endpoint Support

| Endpoint | OpenAI | Anthropic | Google | Azure | Bedrock | Ollama |
|----------|:------:|:---------:|:------:|:-----:|:-------:|:------:|
| Chat completions | yes | yes | yes | yes | yes | yes |
| Streaming | yes | yes | yes | yes | yes | yes |
| Embeddings | yes | — | — | yes | — | yes |
| Image generation | yes | — | — | yes | — | — |
| Audio speech | yes | — | — | yes | — | — |
| Audio transcription | yes | — | — | yes | — | — |
| Tool/function calling | yes | yes | yes | yes | yes | yes |

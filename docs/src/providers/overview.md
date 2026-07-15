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
| `deepseek` | DeepSeek models | Pass-through (OpenAI + native Anthropic) |
| `zai` | z.ai GLM models | Pass-through (OpenAI + native Anthropic) |
| `qwen` | Alibaba Qwen (DashScope) models | Pass-through (OpenAI + native Anthropic) |
| `minimax` | MiniMax models | Pass-through (OpenAI + native Anthropic) |
| `kimi` | Moonshot Kimi models | Pass-through (OpenAI + native Anthropic) |

The last five (`deepseek` … `kimi`) share one **compat** implementation — each is
just a name plus two base URLs in the provider crate's compat table. Adding
another OpenAI+Anthropic provider is a one-line entry there.

The proxy is **dialect-pure**: each endpoint forwards raw bytes only to providers
that natively speak that dialect, with no format translation. So an OpenAI-only
provider (e.g. xAI Grok or Meta, configured as `kind = "openai"` with a
`base_url`) is reachable through `/v1/chat/completions` but **not** through the
`/v1/messages` Anthropic endpoint. To serve Anthropic-format traffic, a provider
needs a native Anthropic endpoint — that's what the compat table is for.

## Common Fields

Every provider supports these fields:

```toml
[providers.name]
kind = "..."           # optional — defaults to the section name (`name`)
api_key = "..."        # API key (supports ${ENV_VAR})
base_url = "..."       # base URL override
models = ["..."]       # model names this provider serves
weight = 1             # routing weight (higher = more traffic)
max_retries = 2        # retries on transient errors (429, 5xx)
timeout = 30           # per-request timeout in seconds
```

When `kind` is omitted, the section name is used as the kind — `[providers.zai]`
resolves to kind `zai`. Set `kind` explicitly only when the section name isn't
the kind, e.g. running two `openai` instances under different names for routing.

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

The `Compat` column covers every compat-table provider (`deepseek`, `zai`,
`qwen`, `minimax`, `kimi`) — they share one implementation, so their endpoint
support is identical. Whether a given provider actually serves embeddings
varies; see its page.

| Endpoint | OpenAI | Anthropic | Google | Azure | Bedrock | Ollama | Compat |
|----------|:------:|:---------:|:------:|:-----:|:-------:|:------:|:------:|
| Chat completions | yes | yes | yes | yes | yes | yes | yes |
| Streaming | yes | yes | yes | yes | yes | yes | yes |
| Embeddings | yes | — | — | yes | — | yes | yes |
| Image generation | yes | — | — | yes | — | — | — |
| Audio speech | yes | — | — | yes | — | — | — |
| Audio transcription | yes | — | — | yes | — | — | — |
| Anthropic Messages | — | yes | — | — | — | — | yes |
| Tool/function calling | yes | yes | yes | yes | yes | yes | yes |

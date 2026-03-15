# crabtalk-provider

Provider dispatch for the [crabtalk](https://github.com/clearloop/crabtalk) LLM API gateway.

Routes requests to upstream LLM providers via the `Provider` enum:

- **OpenAI-compatible**: Pass-through for OpenAI, Azure, Ollama, vLLM, Groq (URL + auth rewrite, body forwarded as-is)
- **Anthropic**: Translates OpenAI format to/from the Anthropic Messages API
- **Google / Bedrock**: Defined but not yet implemented

Also provides `ProviderRegistry` for model-name → provider lookup from config.

## License

MIT OR Apache-2.0

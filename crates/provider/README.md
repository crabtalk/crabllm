# crabtalk-provider

[![crates.io][badge]][crate]

Provider dispatch for the [crabtalk](https://github.com/clearloop/crabtalk) LLM API gateway.

Routes requests to upstream LLM providers via the `Provider` enum:

- **OpenAI-compatible** — pass-through for OpenAI, Ollama, vLLM, Groq, Together (URL + auth rewrite)
- **Anthropic** — full translation to/from the Messages API, including tool calling and streaming
- **Google Gemini** — full translation to/from the Gemini API
- **Azure OpenAI** — deployment-based URL rewrite with `api-key` header auth
- **AWS Bedrock** — Converse API with SigV4 signing (`provider-bedrock` feature)

Also provides `ProviderRegistry` for model-name lookup, weighted routing, and
model aliasing.

## License

MIT OR Apache-2.0

[badge]: https://img.shields.io/crates/v/crabtalk-provider.svg
[crate]: https://crates.io/crates/crabtalk-provider

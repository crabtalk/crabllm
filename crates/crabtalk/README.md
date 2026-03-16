# crabtalk

[![crates.io][badge]][crate]

High-performance LLM API gateway in Rust.

Crabtalk exposes an OpenAI-compatible API and routes requests to the configured
provider — OpenAI, Anthropic, Google Gemini, Azure, AWS Bedrock, Ollama, and any
OpenAI-compatible service.

## Install

```bash
cargo install crabtalk
```

## Usage

```bash
crabtalk --config crabtalk.toml
```

See the [docs](https://clearloop.github.io/crabtalk) for configuration,
providers, routing, and extensions.

## License

MIT OR Apache-2.0

[badge]: https://img.shields.io/crates/v/crabtalk.svg
[crate]: https://crates.io/crates/crabtalk

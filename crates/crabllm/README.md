# crabllm

[![crates.io][badge]][crate]

High-performance LLM API gateway in Rust.

Crabllm exposes an OpenAI-compatible API and routes requests to the configured
provider — OpenAI, Anthropic, Google Gemini, Azure, Ollama, and any
OpenAI-compatible service.

## Install

```bash
cargo install crabllm
```

## Usage

```bash
crabllm --config crabllm.toml
```

See the [docs](https://clearloop.github.io/crabllm) for configuration,
providers, routing, and extensions.

## License

MIT OR Apache-2.0

[badge]: https://img.shields.io/crates/v/crabllm.svg
[crate]: https://crates.io/crates/crabllm

# crabtalk

High-performance LLM API gateway in Rust.

Crabtalk exposes an OpenAI-compatible API and routes requests to the configured
provider — OpenAI, Anthropic, Azure, Ollama, and any OpenAI-compatible service.

## Install

```bash
cargo install crabtalk
```

## Usage

```bash
crabtalk --config crabtalk.toml
```

See the [repository](https://github.com/clearloop/crabtalk) for full
documentation and configuration examples.

## License

MIT OR Apache-2.0

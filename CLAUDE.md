See [CONTRIBUTING.md](CONTRIBUTING.md) for codebase architecture.

## Provider crate has external consumers

The provider crate is a public library — other products depend on it directly,
not just the proxy. Never remove methods from the `Provider` trait (e.g.
`anthropic_messages_stream`) just because the proxy doesn't call them.

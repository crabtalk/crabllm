See [CONTRIBUTING.md](CONTRIBUTING.md) for codebase architecture.

## Translate, don't validate

Reshaping a request between wire formats is the provider/core type layer's job.
Judging whether the upstream will accept the result is not — no capability
gates, no hand-maintained model allowlists, no clamping a value into a range we
guessed at. When the reason for adding a check is "otherwise the provider
400s", that 400 is the answer the caller wants, and a model list goes stale the
day the next model ships.

## Provider crate has external consumers

The provider crate is a public library — other products depend on it directly,
not just the proxy. Never remove methods from the `Provider` trait (e.g.
`anthropic_messages_stream`) just because the proxy doesn't call them.

# crabllm-sdk

Typed Rust client for a [crabllm](https://github.com/crabtalk/crabllm) LLM API
gateway (a deployed `crabllm-proxy`).

`Client` talks HTTP to the gateway and implements `crabllm_core::Provider`, so
it composes with `crabllm_core::Retrying` and drops into any `Provider`-generic
code. Streaming responses are parsed by the shared `crabllm_core::codec` — the
same code the gateway uses, so client and server never drift.

```rust
use crabllm_sdk::{Client, core::{Provider, Retrying}};

let client = Retrying::new(Client::new("https://gateway.example.com", "sk-..."));
let resp = client.anthropic_messages(&request).await?;
```

## TLS

Mutually pick one backend (mirrors `crabllm-provider`):

```sh
cargo build -p crabllm-sdk                                          # native-tls (default)
cargo build -p crabllm-sdk --no-default-features --features rustls  # rustls
```

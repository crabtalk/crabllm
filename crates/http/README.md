# crabllm-http

Outbound HTTP client for the [crabllm](https://github.com/crabtalk/crabllm)
workspace, shared by `crabllm-provider` (server-side dispatch to upstreams) and
`crabllm-sdk` (client to a gateway).

One backend, chosen at compile time and mutually exclusive:

- `hyper` (default) — low-level hyper-util client.
- `reqwest` — reqwest client.

TLS is likewise mutually exclusive: `native-tls` (default) or `rustls`.

```sh
cargo build -p crabllm-http                                          # hyper + native-tls
cargo build -p crabllm-http --no-default-features --features "reqwest,rustls"
```

The client adds no business logic: it POSTs/GETs bytes, surfaces status, and
streams response bodies, mapping transport failures to `crabllm_core::Error`.

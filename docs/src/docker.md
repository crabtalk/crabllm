# Running with Docker

The `crabllm` image is published to the GitHub Container Registry as `ghcr.io/crabtalk/crabllm:latest`. It's a minimal `debian:bookworm-slim` image with the statically-linked binary inside.

## Quick start

```bash
mkdir -p ./crabllm-data
docker run -d --name crabllm \
  -p 5632:5632 \
  -v ./crabllm-data:/data \
  ghcr.io/crabtalk/crabllm:latest \
  serve --config /data/crabllm.toml --bind 0.0.0.0:5632
```

First run: `crabllm.toml` doesn't exist, so the server generates one in `./crabllm-data/crabllm.toml` with a fresh admin token and default API key. Inspect it:

```bash
cat ./crabllm-data/crabllm.toml | grep -E 'admin_token|key'
```

Copy the two `sk-…` values — you need the admin token to manage providers, and the default key to call the gateway.

## Connecting with crabctl

Install `crabctl` on the host (or any machine with network access to the container):

```bash
cargo install crabctl
```

Export the admin token and point it at the container:

```bash
export CRABLLM_URL=http://127.0.0.1:5632
export CRABLLM_TOKEN=<admin_token-from-crabllm.toml>
```

Add a provider dynamically (no restart, no TOML edits):

```bash
# Known kind — OpenAI
crabctl providers create openai \
  --kind openai \
  --api-key "$OPENAI_API_KEY"

# Self-defined kind for any OpenAI-compatible upstream
crabctl providers create openrouter \
  --kind openrouter \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "$OPENROUTER_API_KEY"
```

With `--models` omitted, the server calls `{base_url}/models` and populates the list automatically.

List what's live:

```bash
crabctl providers list
crabctl keys list
```

## Calling the gateway

Use the default key (not the admin token) for completions:

```bash
curl http://127.0.0.1:5632/v1/chat/completions \
  -H "Authorization: Bearer <default_key-from-crabllm.toml>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

OpenAPI docs live at `http://127.0.0.1:5632/docs`. An online snapshot is hosted at [crabtalk.github.io/crabllm/api](https://crabtalk.github.io/crabllm/api).

## Passing upstream API keys via environment

`crabllm.toml` expands `${VAR}` at load time, so secrets don't have to live in the file. Edit the generated config:

```toml
[providers.openai]
kind = "openai"
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]
```

Then pass the variable to the container:

```bash
docker run -d --name crabllm \
  -p 5632:5632 \
  -v ./crabllm-data:/data \
  -e OPENAI_API_KEY \
  ghcr.io/crabtalk/crabllm:latest \
  serve --config /data/crabllm.toml --bind 0.0.0.0:5632
```

## docker-compose

```yaml
services:
  crabllm:
    image: ghcr.io/crabtalk/crabllm:latest
    restart: unless-stopped
    ports:
      - "5632:5632"
    volumes:
      - ./crabllm-data:/data
    environment:
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
    command:
      - serve
      - --config
      - /data/crabllm.toml
      - --bind
      - 0.0.0.0:5632
```

## Verbose logs

Pass `-v` (info), `-vv` (debug), or `-vvv` (trace) before the subcommand to see request/response lines and outbound provider calls:

```bash
docker run --rm ghcr.io/crabtalk/crabllm:latest -vv serve --config /data/crabllm.toml
```

`RUST_LOG` is honored when no `-v` is given.

## TLS backends

The published image uses the `native-tls` default, which links `libssl3` at runtime (installed in the image). If you need the pure-Rust `rustls` stack — for example to drop `libssl` from a derived image — rebuild from source:

```bash
cargo install crabllm --no-default-features --features rustls,openapi
```

# AWS Bedrock

The `bedrock` provider translates requests to the AWS Bedrock Converse API with
SigV4 request signing. No AWS SDK dependency — signing is handled internally.

## Feature Flag

Bedrock support requires the `provider-bedrock` cargo feature:

```bash
cargo install crabtalk --features provider-bedrock
```

## Configuration

```toml
[providers.bedrock]
kind = "bedrock"
region = "us-east-1"
access_key = "${AWS_ACCESS_KEY_ID}"
secret_key = "${AWS_SECRET_ACCESS_KEY}"
models = ["anthropic.claude-3-5-sonnet-20241022-v2:0"]
```

## Translation

- **System messages** — mapped to the Bedrock `system` field.
- **Tool calling** — tool definitions mapped to `toolConfig.tools[].toolSpec`,
  tool results to `toolResult` content blocks.
- **Stop reasons** — `end_turn` to `stop`, `tool_use` to `tool_calls`,
  `max_tokens` to `length`.
- **Streaming** — uses ConverseStream with AWS event-stream binary framing.

## Usage

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Limitations

- Embeddings, image generation, and audio endpoints are not supported.

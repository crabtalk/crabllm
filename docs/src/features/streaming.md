# Streaming

Crabtalk supports Server-Sent Events (SSE) streaming for chat completions across
all providers. Streams are proxied without buffering — tokens arrive
incrementally as the provider generates them.

## Usage

Set `"stream": true` in the request body:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a haiku."}],
    "stream": true
  }'
```

The response is a stream of SSE events:

```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"An"}}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":" old"}}]}

data: [DONE]
```

## Provider Translation

For non-OpenAI providers, crabtalk translates the provider's native streaming
format to OpenAI-compatible SSE chunks:

- **Anthropic** — `message_start`, `content_block_delta` events translated to
  `chat.completion.chunk` format.
- **Google Gemini** — `streamGenerateContent` response parts translated to
  OpenAI chunks.
- **Bedrock** — AWS event-stream binary frames decoded and translated.
- **Azure** — same SSE format as OpenAI, no translation needed.

## Extension Hooks

Extensions can observe each streaming chunk via the `on_chunk` hook. The rate
limiter and budget extension use this to count tokens in real-time as they arrive.

## Keep-Alive

SSE connections include automatic keep-alive pings to prevent proxy/load balancer
timeouts during long generation pauses.

## Error Handling

If an error occurs mid-stream (after the first chunk has been sent), it is
delivered as an SSE event with an error payload. The stream then terminates.
Retry and fallback only apply before the stream starts.

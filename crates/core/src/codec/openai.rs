use crate::{ByteStream, ChatCompletionChunk, Error};
use bytes::{Buf, BytesMut};
use futures::stream::{self, Stream};

/// Parse an SSE byte stream into `ChatCompletionChunk` items.
pub fn sse_stream(
    byte_stream: ByteStream,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    stream::unfold(
        (byte_stream, BytesMut::new()),
        |(mut byte_stream, mut buffer)| async move {
            use futures::StreamExt;

            loop {
                if let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                    let mut line_end = newline_pos;
                    if line_end > 0 && buffer[line_end - 1] == b'\r' {
                        line_end -= 1;
                    }
                    let line = &buffer[..line_end];

                    if line.is_empty() {
                        buffer.advance(newline_pos + 1);
                        continue;
                    }

                    if let Some(data) = line.strip_prefix(b"data: ") {
                        let data = match std::str::from_utf8(data) {
                            Ok(s) => s.trim(),
                            Err(_) => {
                                buffer.advance(newline_pos + 1);
                                continue;
                            }
                        };
                        if data == "[DONE]" {
                            return None;
                        }
                        let result = match crate::json::from_str::<ChatCompletionChunk>(data) {
                            Ok(chunk) => Ok(chunk),
                            Err(e) => Err(Error::Decode(format!("SSE parse error: {e}"))),
                        };
                        buffer.advance(newline_pos + 1);
                        return Some((result, (byte_stream, buffer)));
                    }
                    // Skip non-data lines (comments, event:, etc.)
                    buffer.advance(newline_pos + 1);
                    continue;
                }

                // Need more data from the stream.
                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.extend_from_slice(&bytes);
                    }
                    Some(Err(e)) => {
                        return Some((Err(Error::Network(e.to_string())), (byte_stream, buffer)));
                    }
                    None => return None,
                }
            }
        },
    )
}

use crate::{ByteStream, Error};
use bytes::{Buf, BytesMut};
use futures::stream::{self, Stream, StreamExt};

/// Frame an SSE byte stream into successive `data:` payloads.
///
/// Yields each `data:` field value, trimmed and owned. Blank lines, comments,
/// and other fields (`event:`, `id:`, …) are skipped; a `data: [DONE]` sentinel
/// ends the stream. Transport failures surface as [`Error::Network`]. This is
/// the framing every provider's SSE parser shares — the per-format decoding
/// (which JSON type a payload becomes, and any cross-event state) lives in the
/// individual codec modules.
pub(crate) fn data_lines(byte_stream: ByteStream) -> impl Stream<Item = Result<String, Error>> {
    stream::unfold(
        (byte_stream, BytesMut::new()),
        |(mut byte_stream, mut buffer)| async move {
            loop {
                if let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                    let mut line_end = newline_pos;
                    if line_end > 0 && buffer[line_end - 1] == b'\r' {
                        line_end -= 1;
                    }
                    let payload = buffer[..line_end]
                        .strip_prefix(b"data: ")
                        .and_then(|d| std::str::from_utf8(d).ok())
                        .map(|s| s.trim().to_string());
                    buffer.advance(newline_pos + 1);
                    match payload {
                        Some(p) if p == "[DONE]" => return None,
                        Some(p) => return Some((Ok(p), (byte_stream, buffer))),
                        None => continue, // blank, comment, or non-`data:` line
                    }
                }

                match byte_stream.next().await {
                    Some(Ok(bytes)) => buffer.extend_from_slice(&bytes),
                    Some(Err(e)) => {
                        return Some((Err(Error::Network(e.to_string())), (byte_stream, buffer)));
                    }
                    None => return None,
                }
            }
        },
    )
}

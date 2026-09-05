use crate::{ByteStream, ChatCompletionChunk, Error};
use alloc::format;
use futures_util::stream::{Stream, StreamExt};

/// Parse an SSE byte stream into `ChatCompletionChunk` items. Each `data:`
/// payload is one chunk; the `[DONE]` sentinel ends the stream.
pub fn sse_stream(
    byte_stream: ByteStream,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    crate::codec::sse::data_lines(byte_stream).map(|line| {
        crate::json::from_str::<ChatCompletionChunk>(&line?)
            .map_err(|e| Error::Decode(format!("SSE parse error: {e}")))
    })
}

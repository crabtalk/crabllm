use bytes::{Bytes, BytesMut};
use crabllm_core::ByteStream;
use futures::StreamExt;
use http_body_util::{BodyExt, BodyStream};

#[derive(serde::Deserialize)]
struct Peek {
    model: String,
    #[serde(default)]
    stream: Option<bool>,
}

pub struct RequestBody {
    buf: BytesMut,
    rest: axum::body::Body,
    pub model: String,
    pub is_stream: bool,
}

impl RequestBody {
    pub async fn read(mut body: axum::body::Body) -> Option<Self> {
        let mut buf = BytesMut::with_capacity(1024);
        loop {
            if let Ok(peek) = crabllm_core::json::from_slice::<Peek>(&buf) {
                return Some(Self::from_peek(buf, body, peek));
            }
            let frame = body.frame().await?.ok()?;
            if let Some(data) = frame.data_ref() {
                buf.extend_from_slice(data);
            }
            if buf.len() > 64 * 1024 {
                break;
            }
        }
        let peek = crabllm_core::json::from_slice::<Peek>(&buf).ok()?;
        Some(Self::from_peek(buf, body, peek))
    }

    fn from_peek(buf: BytesMut, rest: axum::body::Body, peek: Peek) -> Self {
        Self {
            buf,
            rest,
            model: peek.model,
            is_stream: peek.stream == Some(true),
        }
    }

    pub fn into_stream(self) -> ByteStream {
        let prefix = inject_stream_options(self.buf.freeze());
        let prefix_once = futures::stream::once(async { Ok::<_, std::io::Error>(prefix) });
        let rest = BodyStream::new(self.rest).filter_map(|f| {
            std::future::ready(match f {
                Ok(f) => f.into_data().ok().map(Ok),
                Err(e) => Some(Err(std::io::Error::other(e))),
            })
        });
        Box::pin(prefix_once.chain(rest))
    }

    pub async fn into_bytes(self) -> Option<Bytes> {
        let remaining = self.rest.collect().await.ok()?.to_bytes();
        let mut full = BytesMut::with_capacity(self.buf.len() + remaining.len());
        full.extend_from_slice(&self.buf);
        full.extend_from_slice(&remaining);
        Some(full.freeze())
    }
}

fn inject_stream_options(prefix: Bytes) -> Bytes {
    if prefix
        .windows(b"\"stream_options\"".len())
        .any(|w| w == b"\"stream_options\"")
    {
        return prefix;
    }
    let Some(brace) = prefix.iter().position(|&b| b == b'{') else {
        return prefix;
    };
    let injection = b"\"stream_options\":{\"include_usage\":true},";
    let mut patched = Vec::with_capacity(prefix.len() + injection.len());
    patched.extend_from_slice(&prefix[..=brace]);
    patched.extend_from_slice(injection);
    patched.extend_from_slice(&prefix[brace + 1..]);
    Bytes::from(patched)
}

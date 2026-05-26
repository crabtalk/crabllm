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

pub enum ReadError {
    Io(String),
    InvalidJson(String),
}

pub struct RequestBody {
    buf: BytesMut,
    rest: axum::body::Body,
    pub model: String,
    pub is_stream: bool,
}

const PREFIX_BUDGET: usize = 64 * 1024;

impl RequestBody {
    pub async fn read(mut body: axum::body::Body) -> Result<Self, ReadError> {
        let mut buf = BytesMut::with_capacity(1024);
        loop {
            if let Ok(peek) = crabllm_core::json::from_slice::<Peek>(&buf) {
                return Ok(Self::from_peek(buf, body, peek));
            }
            let Some(frame) = body.frame().await else {
                break;
            };
            let frame = frame.map_err(|e| ReadError::Io(e.to_string()))?;
            if let Some(data) = frame.data_ref() {
                buf.extend_from_slice(data);
            }
            if buf.len() > PREFIX_BUDGET {
                break;
            }
        }

        if crabllm_core::json::from_slice::<Peek>(&buf).is_err() {
            let remaining = body
                .collect()
                .await
                .map_err(|e| ReadError::Io(e.to_string()))?
                .to_bytes();
            buf.extend_from_slice(&remaining);
            body = axum::body::Body::empty();
        }

        let peek = crabllm_core::json::from_slice::<Peek>(&buf)
            .map_err(|e| ReadError::InvalidJson(e.to_string()))?;
        Ok(Self::from_peek(buf, body, peek))
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

    pub async fn into_bytes(self) -> Result<Bytes, ReadError> {
        let remaining = self
            .rest
            .collect()
            .await
            .map_err(|e| ReadError::Io(e.to_string()))?
            .to_bytes();
        let mut full = BytesMut::with_capacity(self.buf.len() + remaining.len());
        full.extend_from_slice(&self.buf);
        full.extend_from_slice(&remaining);
        Ok(full.freeze())
    }
}

fn inject_stream_options(prefix: Bytes) -> Bytes {
    if let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&prefix)
        && let Some(obj) = val.as_object_mut()
    {
        obj.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
        if let Ok(out) = serde_json::to_vec(&val) {
            return Bytes::from(out);
        }
    }

    // Truncated prefix — inject after `{` without stripping.
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

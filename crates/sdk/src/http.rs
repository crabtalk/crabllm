use bytes::Bytes;
use crabllm_core::{ByteStream, Error};
use futures::StreamExt;
use std::time::Duration;

/// Minimal reqwest wrapper for talking to a crabllm gateway. No redirects.
/// Maps transport failures to [`Error::Network`] and upstream 4xx/5xx to
/// [`Error::Provider`] (carrying the gateway's own response body verbatim).
#[derive(Clone, Debug)]
pub(crate) struct Http {
    inner: reqwest::Client,
}

impl Http {
    pub(crate) fn new() -> Self {
        let inner = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("crabllm-sdk: failed to build reqwest client");
        Self { inner }
    }

    /// POST a body and return the full response bytes.
    pub(crate) async fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<Bytes, Error> {
        let resp = self.send(url, headers, body).await?;
        let status = resp.status().as_u16();
        let retry_after = retry_after(&resp);
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::Network(e.to_string()))?;
        if status >= 400 {
            return Err(Error::Provider {
                status,
                body: String::from_utf8_lossy(&bytes).into_owned(),
                retry_after,
            });
        }
        Ok(bytes)
    }

    /// POST a body and stream the raw response. A 4xx/5xx is drained and
    /// returned as [`Error::Provider`] before any streaming begins.
    pub(crate) async fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        let resp = self.send(url, headers, body).await?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let retry_after = retry_after(&resp);
            let bytes = resp
                .bytes()
                .await
                .map_err(|e| Error::Network(e.to_string()))?;
            return Err(Error::Provider {
                status,
                body: String::from_utf8_lossy(&bytes).into_owned(),
                retry_after,
            });
        }
        let stream = resp.bytes_stream().map(|r| r.map_err(std::io::Error::other));
        Ok(Box::pin(stream))
    }

    async fn send(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<reqwest::Response, Error> {
        let mut req = self.inner.post(url).body(body);
        for &(name, value) in headers {
            req = req.header(name, value);
        }
        req.send().await.map_err(|e| Error::Network(e.to_string()))
    }
}

/// Parse a `retry-after` header (integer seconds only) into a `Duration`.
fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

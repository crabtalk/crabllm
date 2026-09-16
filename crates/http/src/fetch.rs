use crate::{ByteStream, RawResponse, parse_retry_after};
use bytes::{Bytes, BytesMut};
use crabllm_core::{Error, SingleThreaded};
use futures::stream::StreamExt;
use std::future::Future;
use web_time::Instant;

/// `fetch` client. The browser dials the connection, so TLS, redirects, the
/// cookie jar and which origins are reachable at all (CORS) are its decisions,
/// not ours.
///
/// No timeout is set here. `fetch` offers only a total-request abort, which is
/// the wrong shape for a completion that streams for minutes; the per-attempt
/// timeout and stream-idle bound in [`Retrying`](crabllm_core::Retrying) — on
/// by default in `crabllm-sdk` — are the ones that fit.
///
/// Every method wraps its whole body in [`SingleThreaded`] rather than just the
/// `await` points: a `reqwest::Response` holds the abort closure `fetch` needs
/// to cancel, and an `async fn` stores its arguments whether or not they live
/// across an `await`, so passing one anywhere would sink the `Send` bound
/// [`Provider`](crabllm_core::Provider) asks for.
#[derive(Clone, Debug)]
pub struct HttpClient {
    inner: SingleThreaded<reqwest::Client>,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            inner: SingleThreaded(reqwest::Client::new()),
        }
    }

    pub fn get(
        &self,
        url: &str,
        headers: &[(&str, &str)],
    ) -> impl Future<Output = Result<RawResponse, Error>> + Send {
        SingleThreaded(async move {
            let start = Instant::now();
            let mut req = self.inner.get(url);
            for &(name, value) in headers {
                req = req.header(name, value);
            }
            let resp = req.send().await.map_err(|e| {
                tracing::debug!(url, error = %e, "provider GET failed");
                Error::Network(e.to_string())
            })?;
            let raw = collect(resp).await?;
            tracing::debug!(
                url,
                status = raw.status,
                response_bytes = raw.body.len(),
                latency_ms = start.elapsed().as_millis() as u64,
                "provider GET"
            );
            Ok(raw)
        })
    }

    pub fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> impl Future<Output = Result<RawResponse, Error>> + Send {
        SingleThreaded(async move {
            let request_bytes = body.len();
            let start = Instant::now();
            let resp = self.send(url, headers, body).await?;
            let raw = collect(resp).await?;
            tracing::debug!(
                url,
                status = raw.status,
                request_bytes,
                response_bytes = raw.body.len(),
                latency_ms = start.elapsed().as_millis() as u64,
                "provider call"
            );
            Ok(raw)
        })
    }

    /// `fetch` cannot stream a request body — upload streaming needs a duplex
    /// request no browser exposes to wasm — so the stream is drained into
    /// memory first. Callers that only stream the *response*, which is every
    /// chat call, are unaffected.
    pub fn post_stream_body(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        mut body_stream: ByteStream,
    ) -> impl Future<Output = Result<ByteStream, Error>> + Send {
        SingleThreaded(async move {
            let mut body = BytesMut::new();
            while let Some(chunk) = body_stream.next().await {
                body.extend_from_slice(&chunk?);
            }
            self.post_stream(url, headers, body.freeze()).await
        })
    }

    pub fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> impl Future<Output = Result<ByteStream, Error>> + Send {
        SingleThreaded(async move {
            let start = Instant::now();
            let resp = self.send(url, headers, body).await?;

            let status = resp.status().as_u16();
            if status >= 400 {
                let raw = collect(resp).await?;
                tracing::debug!(
                    url,
                    status,
                    latency_ms = start.elapsed().as_millis() as u64,
                    "provider stream error"
                );
                return Err(Error::Provider {
                    status,
                    body: String::from_utf8_lossy(&raw.body).into_owned(),
                    retry_after: raw.retry_after,
                });
            }

            tracing::debug!(
                url,
                status,
                ttfb_ms = start.elapsed().as_millis() as u64,
                "provider stream opened"
            );

            let chunks = SingleThreaded(resp.bytes_stream())
                .map(|chunk| chunk.map_err(|e| Error::Network(e.to_string())));
            Ok(Box::pin(chunks) as ByteStream)
        })
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
        req.send().await.map_err(|e| {
            tracing::debug!(url, error = %e, "provider call failed");
            Error::Network(e.to_string())
        })
    }
}

/// Read a whole response into a [`RawResponse`].
async fn collect(resp: reqwest::Response) -> Result<RawResponse, Error> {
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after);
    let body = resp
        .bytes()
        .await
        .map_err(|e| Error::Network(e.to_string()))?;
    Ok(RawResponse {
        status,
        body,
        content_type,
        retry_after,
    })
}

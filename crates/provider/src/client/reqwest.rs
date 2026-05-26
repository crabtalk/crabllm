use crate::client::{ByteStream, RawResponse, parse_retry_after};
use bytes::Bytes;
use crabllm_core::Error;
use futures::stream::StreamExt;
use std::time::Instant;

/// reqwest client. No redirects, no cookies, no decompression.
#[derive(Clone, Debug)]
pub struct HttpClient {
    inner: reqwest::Client,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    pub fn new() -> Self {
        let builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_gzip()
            .no_brotli()
            .no_deflate();
        #[cfg(not(feature = "http2"))]
        let builder = builder.http1_only();
        let inner = builder
            .build()
            .expect("crabllm: failed to build reqwest client");
        Self { inner }
    }

    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<RawResponse, Error> {
        let start = Instant::now();
        let mut req = self.inner.get(url);
        for &(name, value) in headers {
            req = req.header(name, value);
        }
        let resp = req.send().await.map_err(|e| {
            tracing::debug!(url, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider GET failed");
            Error::Internal(e.to_string())
        })?;
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
            .map_err(|e| Error::Internal(e.to_string()))?;
        tracing::debug!(
            url,
            status,
            response_bytes = body.len(),
            latency_ms = start.elapsed().as_millis() as u64,
            "provider GET"
        );
        Ok(RawResponse {
            status,
            body,
            content_type,
            retry_after,
        })
    }

    pub async fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<RawResponse, Error> {
        let request_bytes = body.len();
        let start = Instant::now();
        let mut req = self.inner.post(url).body(body);
        for &(name, value) in headers {
            req = req.header(name, value);
        }
        let resp = req.send().await.map_err(|e| {
            tracing::debug!(url, request_bytes, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider call failed");
            Error::Internal(e.to_string())
        })?;
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
            .map_err(|e| Error::Internal(e.to_string()))?;
        tracing::debug!(
            url,
            status,
            request_bytes,
            response_bytes = body.len(),
            latency_ms = start.elapsed().as_millis() as u64,
            "provider call"
        );
        Ok(RawResponse {
            status,
            body,
            content_type,
            retry_after,
        })
    }

    pub async fn post_stream_body(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body_stream: ByteStream,
    ) -> Result<ByteStream, Error> {
        self.send_stream(url, headers, reqwest::Body::wrap_stream(body_stream))
            .await
    }

    pub async fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        self.send_stream(url, headers, reqwest::Body::from(body))
            .await
    }

    async fn send_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: reqwest::Body,
    ) -> Result<ByteStream, Error> {
        let start = Instant::now();
        let mut req = self.inner.post(url).body(body);
        for &(name, value) in headers {
            req = req.header(name, value);
        }
        let resp = req.send().await.map_err(|e| {
            tracing::debug!(url, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider stream failed");
            Error::Internal(e.to_string())
        })?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after);
            let body = resp
                .bytes()
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            let text = String::from_utf8_lossy(&body).into_owned();
            tracing::debug!(
                url,
                status,
                latency_ms = start.elapsed().as_millis() as u64,
                "provider stream error"
            );
            return Err(Error::Provider {
                status,
                body: text,
                retry_after,
            });
        }
        tracing::debug!(
            url,
            status,
            ttfb_ms = start.elapsed().as_millis() as u64,
            "provider stream opened"
        );
        let stream = resp
            .bytes_stream()
            .map(|r| r.map_err(std::io::Error::other));
        Ok(Box::pin(stream))
    }
}

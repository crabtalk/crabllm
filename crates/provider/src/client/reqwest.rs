use crate::client::{ByteStream, RawResponse};
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
        })
    }

    pub async fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        let request_bytes = body.len();
        let start = Instant::now();
        let mut req = self.inner.post(url).body(body);
        for &(name, value) in headers {
            req = req.header(name, value);
        }
        let resp = req.send().await.map_err(|e| {
            tracing::debug!(url, request_bytes, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider stream failed");
            Error::Internal(e.to_string())
        })?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let body = resp
                .bytes()
                .await
                .map_err(|e| Error::Internal(e.to_string()))?;
            let text = String::from_utf8_lossy(&body).into_owned();
            tracing::debug!(
                url,
                status,
                request_bytes,
                response_bytes = body.len(),
                latency_ms = start.elapsed().as_millis() as u64,
                "provider stream error"
            );
            return Err(Error::Provider { status, body: text });
        }
        tracing::debug!(
            url,
            status,
            request_bytes,
            ttfb_ms = start.elapsed().as_millis() as u64,
            "provider stream opened"
        );
        let stream = resp
            .bytes_stream()
            .map(|r| r.map_err(std::io::Error::other));
        Ok(Box::pin(stream))
    }
}

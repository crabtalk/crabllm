use crate::{ByteStream, RawResponse, parse_retry_after};
use bytes::Bytes;
use crabllm_core::Error;
use futures::stream::StreamExt;
use std::time::{Duration, Instant};
use wstd::http::{Body, Client, ErrorCode, Method, Request, Response, body::util::BodyStream};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Max idle gap between response frames, enforced by the host as wasi:http's
/// `between-bytes-timeout`.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const CONTENT_TYPE: &str = "content-type";
const RETRY_AFTER: &str = "retry-after";

/// wasi:http client. The host dials the connection, so TLS, redirects and
/// which hosts are reachable are the host's decisions.
#[derive(Clone, Debug)]
pub struct HttpClient {
    inner: Client,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    pub fn new() -> Self {
        let mut inner = Client::new();
        inner.set_connect_timeout(CONNECT_TIMEOUT);
        inner.set_between_bytes_timeout(READ_TIMEOUT);
        Self { inner }
    }

    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<RawResponse, Error> {
        let start = Instant::now();
        let resp = self.send(Method::GET, url, headers, Body::empty()).await?;
        let raw = collect(resp).await?;
        tracing::debug!(
            url,
            status = raw.status,
            response_bytes = raw.body.len(),
            latency_ms = start.elapsed().as_millis() as u64,
            "provider GET"
        );
        Ok(raw)
    }

    pub async fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<RawResponse, Error> {
        let request_bytes = body.len();
        let start = Instant::now();
        let resp = self.send(Method::POST, url, headers, body.into()).await?;
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
    }

    pub async fn post_stream_body(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body_stream: ByteStream,
    ) -> Result<ByteStream, Error> {
        self.send_stream(url, headers, Body::from_try_stream(body_stream))
            .await
    }

    pub async fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        self.send_stream(url, headers, body.into()).await
    }

    async fn send_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Body,
    ) -> Result<ByteStream, Error> {
        let start = Instant::now();
        let resp = self.send(Method::POST, url, headers, body).await?;

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

        let frames = BodyStream::new(resp.into_body().into_boxed_body()).filter_map(|frame| {
            let result = match frame {
                Ok(f) => f.into_data().ok().map(Ok),
                Err(e) => Some(Err(network(e))),
            };
            std::future::ready(result)
        });

        Ok(Box::pin(frames))
    }

    async fn send(
        &self,
        method: Method,
        url: &str,
        headers: &[(&str, &str)],
        body: Body,
    ) -> Result<Response<Body>, Error> {
        let mut builder = Request::builder().method(method).uri(url);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(body)
            .map_err(|e| Error::Internal(e.to_string()))?;

        self.inner.send(req).await.map_err(|e| {
            tracing::debug!(url, error = %e, "provider call failed");
            network(e)
        })
    }
}

/// Read a whole response into a [`RawResponse`].
async fn collect(resp: Response<Body>) -> Result<RawResponse, Error> {
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let retry_after = resp
        .headers()
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after);
    let body = resp.into_body().bytes_contents().await.map_err(network)?;
    Ok(RawResponse {
        status,
        body,
        content_type,
        retry_after,
    })
}

/// wasi:http's timeout codes become [`Error::Timeout`], so a host that gave up
/// on a stalled upstream reads the same as a socket that did.
fn network(e: wstd::http::Error) -> Error {
    match e.downcast_ref::<ErrorCode>() {
        Some(
            ErrorCode::ConnectionTimeout
            | ErrorCode::ConnectionReadTimeout
            | ErrorCode::HttpResponseTimeout,
        ) => Error::Timeout,
        _ => Error::Network(e.to_string()),
    }
}

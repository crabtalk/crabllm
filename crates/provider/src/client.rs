use bytes::Bytes;
use crabllm_core::Error;
use futures::stream::{Stream, StreamExt};
use http_body_util::{BodyExt, BodyStream, Full};
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use std::{pin::Pin, time::Instant};

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

#[cfg(feature = "rustls")]
type Connector = hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;
#[cfg(feature = "native-tls")]
type Connector = hyper_tls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;
type HyperClient = Client<Connector, Full<Bytes>>;

/// Minimal HTTP client for proxy workloads. Wraps hyper-util's pooled
/// client with HTTPS (rustls) and HTTP/2 support. No redirects, no
/// cookies, no decompression — just POST and read.
///
/// Cloning is cheap (`Client` is internally `Arc`-shared).
#[derive(Clone, Debug)]
pub struct HttpClient {
    inner: HyperClient,
}

/// Raw HTTP response — status + body bytes + optional content-type.
pub struct RawResponse {
    pub status: u16,
    pub body: Bytes,
    pub content_type: Option<String>,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    /// Build a new client with TCP_NODELAY, TLS (rustls or native-tls,
    /// feature-gated), and HTTP/2.
    pub fn new() -> Self {
        let https = Self::connector();
        let inner = Client::builder(TokioExecutor::new()).build(https);
        Self { inner }
    }

    #[cfg(feature = "rustls")]
    fn connector() -> Connector {
        hyper_rustls::HttpsConnectorBuilder::new()
            .with_native_roots()
            .expect("crabllm: failed to load native TLS roots")
            .https_or_http()
            .enable_http1()
            .enable_http2()
            .build()
    }

    #[cfg(feature = "native-tls")]
    fn connector() -> Connector {
        let mut http = hyper_util::client::legacy::connect::HttpConnector::new();
        http.enforce_http(false);
        let tls = native_tls_crate::TlsConnector::builder()
            .request_alpns(&["h2", "http/1.1"])
            .build()
            .expect("crabllm: failed to build native TLS connector");
        hyper_tls::HttpsConnector::from((http, tls.into()))
    }

    /// GET a URL and collect the full response.
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<RawResponse, Error> {
        let uri: http::Uri = url
            .parse()
            .map_err(|e: http::uri::InvalidUri| Error::Internal(e.to_string()))?;

        let start = Instant::now();
        let mut builder = http::Request::builder().method(http::Method::GET).uri(uri);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(Full::new(Bytes::new()))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self.inner.request(req).await.map_err(|e| {
            tracing::debug!(
                url,
                latency_ms = elapsed_ms(start),
                error = %e,
                "provider GET failed"
            );
            Error::Internal(e.to_string())
        })?;

        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let body = resp
            .into_body()
            .collect()
            .await
            .map_err(|e| Error::Internal(e.to_string()))?
            .to_bytes();

        tracing::debug!(
            url,
            status,
            response_bytes = body.len(),
            latency_ms = elapsed_ms(start),
            "provider GET"
        );

        Ok(RawResponse {
            status,
            body,
            content_type,
        })
    }

    /// POST a body and collect the full response.
    pub async fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<RawResponse, Error> {
        let uri: http::Uri = url
            .parse()
            .map_err(|e: http::uri::InvalidUri| Error::Internal(e.to_string()))?;

        let request_bytes = body.len();
        let start = Instant::now();

        let mut builder = http::Request::builder().method(http::Method::POST).uri(uri);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(Full::new(body))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self.inner.request(req).await.map_err(|e| {
            tracing::debug!(
                url,
                request_bytes,
                latency_ms = elapsed_ms(start),
                error = %e,
                "provider call failed"
            );
            Error::Internal(e.to_string())
        })?;

        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let body = resp
            .into_body()
            .collect()
            .await
            .map_err(|e| Error::Internal(e.to_string()))?
            .to_bytes();

        tracing::debug!(
            url,
            status,
            request_bytes,
            response_bytes = body.len(),
            latency_ms = elapsed_ms(start),
            "provider call"
        );

        Ok(RawResponse {
            status,
            body,
            content_type,
        })
    }

    /// POST and return the response body as a [`ByteStream`] for SSE.
    /// Returns error on 4xx/5xx.
    pub async fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        let uri: http::Uri = url
            .parse()
            .map_err(|e: http::uri::InvalidUri| Error::Internal(e.to_string()))?;

        let request_bytes = body.len();
        let start = Instant::now();

        let mut builder = http::Request::builder().method(http::Method::POST).uri(uri);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(Full::new(body))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self.inner.request(req).await.map_err(|e| {
            tracing::debug!(
                url,
                request_bytes,
                latency_ms = elapsed_ms(start),
                error = %e,
                "provider stream failed"
            );
            Error::Internal(e.to_string())
        })?;

        let status = resp.status().as_u16();
        if status >= 400 {
            let body = resp
                .into_body()
                .collect()
                .await
                .map_err(|e| Error::Internal(e.to_string()))?
                .to_bytes();
            let text = String::from_utf8_lossy(&body).into_owned();
            tracing::debug!(
                url,
                status,
                request_bytes,
                response_bytes = body.len(),
                latency_ms = elapsed_ms(start),
                "provider stream error"
            );
            return Err(Error::Provider { status, body: text });
        }

        // Latency is time-to-headers, not total stream duration.
        tracing::debug!(
            url,
            status,
            request_bytes,
            ttfb_ms = elapsed_ms(start),
            "provider stream opened"
        );

        Ok(Box::pin(BodyStream::new(resp.into_body()).filter_map(
            |frame| {
                let result = match frame {
                    Ok(f) => f.into_data().ok().map(Ok),
                    Err(e) => Some(Err(std::io::Error::other(e))),
                };
                std::future::ready(result)
            },
        )))
    }
}

/// A boxed byte stream suitable for SSE parsing.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;

use bytes::Bytes;
use crabllm_core::Error;
use futures::stream::{Stream, StreamExt};
use http_body_util::{BodyExt, BodyStream, Full};
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use std::pin::Pin;

type Connector = hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;
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

impl HttpClient {
    /// Build a new client with TCP_NODELAY, TLS (rustls), and HTTP/2.
    pub fn new() -> Self {
        let https = hyper_rustls::HttpsConnectorBuilder::new()
            .with_native_roots()
            .expect("crabllm: failed to load native TLS roots")
            .https_or_http()
            .enable_http1()
            .enable_http2()
            .build();

        let inner = Client::builder(TokioExecutor::new()).build(https);
        Self { inner }
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

        let mut builder = http::Request::builder().method(http::Method::POST).uri(uri);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(Full::new(body))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self
            .inner
            .request(req)
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;

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

        let mut builder = http::Request::builder().method(http::Method::POST).uri(uri);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(Full::new(body))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self
            .inner
            .request(req)
            .await
            .map_err(|e| Error::Internal(e.to_string()))?;

        let status = resp.status().as_u16();
        if status >= 400 {
            let body = resp
                .into_body()
                .collect()
                .await
                .map_err(|e| Error::Internal(e.to_string()))?
                .to_bytes();
            let text = String::from_utf8_lossy(&body).into_owned();
            return Err(Error::Provider { status, body: text });
        }

        Ok(Box::pin(BodyStream::new(resp.into_body()).filter_map(
            |frame| {
                let result = match frame {
                    Ok(f) => f.into_data().ok().map(Ok),
                    Err(e) => Some(Err(std::io::Error::new(std::io::ErrorKind::Other, e))),
                };
                std::future::ready(result)
            },
        )))
    }
}

/// A boxed byte stream suitable for SSE parsing.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;

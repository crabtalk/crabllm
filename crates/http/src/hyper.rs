use crate::{ByteStream, RawResponse, parse_retry_after};
use bytes::Bytes;
use crabllm_core::Error;
use futures::stream::StreamExt;
use http_body::Frame;
use http_body_util::{BodyExt, BodyStream, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use std::time::Instant;

#[cfg(feature = "rustls")]
type Connector = hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;
#[cfg(feature = "native-tls")]
type Connector = hyper_tls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;

type RequestBody = UnsyncBoxBody<Bytes, std::io::Error>;
type HyperClient = Client<Connector, RequestBody>;

fn full_body(bytes: Bytes) -> RequestBody {
    Full::new(bytes)
        .map_err(|never| match never {})
        .boxed_unsync()
}

/// hyper-util client. No redirects, no cookies, no decompression.
#[derive(Clone, Debug)]
pub struct HttpClient {
    inner: HyperClient,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    pub fn new() -> Self {
        let https = Self::connector();
        let inner = Client::builder(TokioExecutor::new()).build(https);
        Self { inner }
    }

    #[cfg(feature = "rustls")]
    fn connector() -> Connector {
        let builder = hyper_rustls::HttpsConnectorBuilder::new()
            .with_native_roots()
            .expect("crabllm: failed to load native TLS roots")
            .https_or_http()
            .enable_http1();
        #[cfg(feature = "http2")]
        let builder = builder.enable_http2();
        builder.build()
    }

    #[cfg(feature = "native-tls")]
    fn connector() -> Connector {
        let mut http = hyper_util::client::legacy::connect::HttpConnector::new();
        http.enforce_http(false);
        #[cfg(feature = "http2")]
        let alpns = ["h2", "http/1.1"];
        #[cfg(not(feature = "http2"))]
        let alpns = ["http/1.1"];
        let tls = native_tls::TlsConnector::builder()
            .request_alpns(&alpns)
            .build()
            .expect("crabllm: failed to build native TLS connector");
        hyper_tls::HttpsConnector::from((http, tls.into()))
    }

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
            .body(full_body(Bytes::new()))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self.inner.request(req).await.map_err(|e| {
            tracing::debug!(url, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider GET failed");
            Error::Network(e.to_string())
        })?;

        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let retry_after = resp
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after);
        let body = resp
            .into_body()
            .collect()
            .await
            .map_err(|e| Error::Network(e.to_string()))?
            .to_bytes();

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
            .body(full_body(body))
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self.inner.request(req).await.map_err(|e| {
            tracing::debug!(url, request_bytes, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider call failed");
            Error::Network(e.to_string())
        })?;

        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let retry_after = resp
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after);
        let body = resp
            .into_body()
            .collect()
            .await
            .map_err(|e| Error::Network(e.to_string()))?
            .to_bytes();

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
        let framed = body_stream.map(|r| r.map(Frame::data));
        let body: RequestBody = BodyExt::boxed_unsync(StreamBody::new(framed));
        self.send_stream(url, headers, body).await
    }

    pub async fn post_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: Bytes,
    ) -> Result<ByteStream, Error> {
        self.send_stream(url, headers, full_body(body)).await
    }

    async fn send_stream(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: RequestBody,
    ) -> Result<ByteStream, Error> {
        let uri: http::Uri = url
            .parse()
            .map_err(|e: http::uri::InvalidUri| Error::Internal(e.to_string()))?;

        let start = Instant::now();
        let mut builder = http::Request::builder().method(http::Method::POST).uri(uri);
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let req = builder
            .body(body)
            .map_err(|e| Error::Internal(e.to_string()))?;

        let resp = self.inner.request(req).await.map_err(|e| {
            tracing::debug!(url, latency_ms = start.elapsed().as_millis() as u64, error = %e, "provider stream failed");
            Error::Network(e.to_string())
        })?;

        let status = resp.status().as_u16();
        if status >= 400 {
            let retry_after = resp
                .headers()
                .get(http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after);
            let body = resp
                .into_body()
                .collect()
                .await
                .map_err(|e| Error::Network(e.to_string()))?
                .to_bytes();
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

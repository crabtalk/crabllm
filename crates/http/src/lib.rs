#[cfg(all(feature = "hyper", feature = "reqwest"))]
compile_error!("crabllm-provider: features `hyper` and `reqwest` are mutually exclusive");

#[cfg(not(any(feature = "hyper", feature = "reqwest")))]
compile_error!("crabllm-provider: enable exactly one of `hyper` or `reqwest`");

#[cfg(all(feature = "native-tls", feature = "rustls"))]
compile_error!("crabllm-provider: features `native-tls` and `rustls` are mutually exclusive");

#[cfg(not(any(feature = "native-tls", feature = "rustls")))]
compile_error!("crabllm-provider: enable exactly one of `native-tls` or `rustls`");

use bytes::Bytes;
use std::time::Duration;

/// Raw HTTP response — status + body bytes + optional content-type.
pub struct RawResponse {
    pub status: u16,
    pub body: Bytes,
    pub content_type: Option<String>,
    pub retry_after: Option<Duration>,
}

impl RawResponse {
    /// Return the response unchanged on a 2xx/3xx status, or map a 4xx/5xx to
    /// [`Error::Provider`](crabllm_core::Error::Provider) carrying the body
    /// verbatim. Lets callers write `client.post(..).await?.error_for_status()?`.
    pub fn error_for_status(self) -> Result<Self, crabllm_core::Error> {
        if self.status >= 400 {
            return Err(crabllm_core::Error::Provider {
                status: self.status,
                body: String::from_utf8_lossy(&self.body).into_owned(),
                retry_after: self.retry_after,
            });
        }
        Ok(self)
    }
}

/// Parse a `retry-after` header value into a Duration.
/// Handles integer seconds only; HTTP-date values are ignored.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

pub use crabllm_core::ByteStream;

#[cfg(feature = "hyper")]
mod hyper;
#[cfg(feature = "hyper")]
pub use hyper::HttpClient;

#[cfg(feature = "reqwest")]
mod reqwest;
#[cfg(feature = "reqwest")]
pub use reqwest::HttpClient;

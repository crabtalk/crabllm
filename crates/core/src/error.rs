use alloc::{
    format,
    string::{String, ToString},
};
use core::time::Duration;
use serde::{Deserialize, Serialize};

/// Shared error type for the crabllm workspace.
///
/// Every variant carries enough structure that retry semantics
/// ([`Error::is_transient`]), HTTP status ([`Error::http_status`]), and the
/// client-facing error `type` ([`Error::kind`]) are derivable from the variant
/// alone — never from string contents. Layers must NOT stringify-and-rewrap an
/// inner error (`Error::Internal(format!("...: {e}"))`); propagate the inner
/// `Error` or classify the cause into the right variant instead.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// TOML config parse error or missing env var. Startup only.
    #[error("config error: {0}")]
    Config(String),

    /// Upstream provider returned (or streamed) an error. `body` is the
    /// upstream's own message, passed through verbatim — never prefixed.
    #[error("provider error (HTTP {status}): {body}")]
    Provider {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
    },

    /// Transport failure talking to the upstream: connect, send, or reading a
    /// (possibly mid-stream) response body. Transient.
    #[error("network error: {0}")]
    Network(String),

    /// An upstream response body could not be deserialized into our types.
    /// Not transient — retrying yields the same unparsable bytes.
    #[error("decode error: {0}")]
    Decode(String),

    /// Our own request could not be serialized. A bug on our side, not the
    /// upstream's. Not transient.
    #[error("encode error: {0}")]
    Encode(String),

    /// The client's request is malformed or violates a precondition (e.g. a
    /// body that doesn't match the schema, or an unsupported option
    /// combination). The client's fault — not transient, 400.
    #[error("invalid request: {0}")]
    Invalid(String),

    /// A provider does not implement the requested operation.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Request could not be routed: unknown model or no matching deployment.
    #[error("routing error: {0}")]
    Routing(String),

    /// Request to upstream provider timed out.
    #[error("request timed out")]
    Timeout,

    /// Genuine catch-all for internal bugs. Not transient — if it happens,
    /// retrying won't help.
    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Whether this error is transient and the request should be retried.
    /// Only network failures, timeouts, and upstream 429/5xx are transient;
    /// everything else (decode, encode, unsupported, routing, internal) is
    /// deterministic and must not be retried.
    pub fn is_transient(&self) -> bool {
        match self {
            Error::Provider { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 504),
            Error::Network(_) | Error::Timeout => true,
            _ => false,
        }
    }

    /// Extract the retry-after duration from a `Provider` error, if present.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Error::Provider { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// HTTP status to return to the client for this error. Single source of
    /// truth for the proxy's error-to-response mapping.
    pub fn http_status(&self) -> u16 {
        match self {
            Error::Provider { status, .. } => *status,
            Error::Network(_) | Error::Decode(_) => 502,
            Error::Unsupported(_) => 501,
            Error::Routing(_) => 404,
            Error::Invalid(_) => 400,
            Error::Timeout => 504,
            Error::Config(_) | Error::Encode(_) | Error::Internal(_) => 500,
        }
    }

    /// OpenAI-compatible error `type` for the client-facing `ApiError`.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Provider { .. } => "upstream_error",
            Error::Network(_) => "network_error",
            Error::Decode(_) => "decode_error",
            Error::Routing(_) | Error::Invalid(_) => "invalid_request_error",
            Error::Unsupported(_) => "unsupported_error",
            Error::Timeout => "timeout_error",
            Error::Config(_) | Error::Encode(_) | Error::Internal(_) => "server_error",
        }
    }

    /// The message to show a user, with the upstream's error envelope peeled
    /// off. A `Provider` body is usually an [`ApiError`] JSON document, whose
    /// own `message` is the only part worth reading; every other variant is
    /// already a bare sentence.
    pub fn message(&self) -> String {
        match self {
            Error::Provider { body, .. } => crate::json::from_str::<ApiError>(body)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| body.clone()),
            other => other.to_string(),
        }
    }

    /// Build an "operation not supported" error for a provider trait method
    /// that has no implementation. Used by `Provider` trait default impls.
    /// Distinct from per-provider rejection messages so log lines can be
    /// disambiguated by grep.
    pub fn not_implemented(method: &str) -> Self {
        Error::Unsupported(format!("provider method '{method}' not implemented"))
    }
}

#[cfg(feature = "gateway")]
impl From<toml::de::Error> for Error {
    fn from(e: toml::de::Error) -> Self {
        Error::Config(e.to_string())
    }
}

/// OpenAI-compatible error response returned to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ApiError {
    pub fn new(message: impl Into<String>, kind: impl Into<String>) -> Self {
        ApiError {
            error: ApiErrorBody {
                message: message.into(),
                kind: kind.into(),
                param: None,
                code: None,
            },
        }
    }
}

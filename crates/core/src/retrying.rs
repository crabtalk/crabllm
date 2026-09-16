use crate::{
    AudioSpeechRequest, BoxStream, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error, ImageRequest,
    MultipartField, Provider, anthropic, gemini,
};
use futures_util::StreamExt;
use std::{future::Future, time::Duration};
#[cfg(not(target_family = "wasm"))]
use tokio::time::{sleep, timeout};

const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const DEFAULT_STREAM_IDLE: Duration = Duration::from_secs(90);
const INITIAL_BACKOFF: Duration = Duration::from_millis(100);

/// A `Provider` wrapper that retries transient failures with exponential
/// backoff and full jitter, and bounds each attempt with a per-call timeout.
///
/// For streaming methods the per-attempt timeout covers only opening the
/// stream. A stall *within* an open stream is bounded separately by
/// `stream_idle`, since a transport without its own read timeout will
/// otherwise wait on a dead upstream forever.
///
/// 429s whose `retry_after` exceeds `max_retry_after` are propagated
/// immediately — the upstream is signalling a wait longer than this wrapper
/// is willing to block for.
#[derive(Debug, Clone)]
pub struct Retrying<P: Provider> {
    inner: P,
    max_retries: u32,
    timeout: Duration,
    max_retry_after: Duration,
    stream_idle: Duration,
}

impl<P: Provider> Retrying<P> {
    /// Wrap a provider with the default retry policy (2 retries, 30s timeout,
    /// 60s max retry-after, 90s stream idle, 100ms initial backoff).
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            max_retries: DEFAULT_MAX_RETRIES,
            timeout: DEFAULT_TIMEOUT,
            max_retry_after: DEFAULT_MAX_RETRY_AFTER,
            stream_idle: DEFAULT_STREAM_IDLE,
        }
    }

    /// Borrow the wrapped provider — e.g. to call inherent (non-`Provider`)
    /// methods on it that the retry wrapper doesn't forward.
    pub fn get_ref(&self) -> &P {
        &self.inner
    }

    /// Override the maximum number of retries. `0` disables retrying — each
    /// call is attempted exactly once (still bounded by the timeout).
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Override the per-attempt timeout. Zero disables the timeout.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    /// Override the maximum `Retry-After` duration this wrapper will honor.
    /// 429s above this threshold are propagated as non-retryable.
    pub fn max_retry_after(mut self, d: Duration) -> Self {
        self.max_retry_after = d;
        self
    }

    /// Override how long an open stream may go without producing a chunk
    /// before it fails with [`Error::Timeout`]. Zero disables the bound.
    pub fn stream_idle(mut self, d: Duration) -> Self {
        self.stream_idle = d;
        self
    }

    /// Whether this error should be retried. Transient errors are retryable
    /// unless they carry a `retry_after` that exceeds the threshold.
    fn should_retry(&self, e: &Error) -> bool {
        if !e.is_transient() {
            return false;
        }
        !matches!(e.retry_after(), Some(ra) if ra > self.max_retry_after)
    }

    async fn timed<T>(
        &self,
        fut: impl Future<Output = Result<T, Error>> + Send,
    ) -> Result<T, Error> {
        if self.timeout.is_zero() {
            return fut.await;
        }
        let Ok(result) = timeout(self.timeout, fut).await else {
            return Err(Error::Timeout);
        };
        result
    }
}

impl<P: Provider> Provider for Retrying<P> {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self.timed(self.inner.chat_completion(request)).await {
                Ok(resp) => return Ok(resp),
                Err(e) if self.should_retry(&e) => {
                    let wait = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    sleep(wait).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self.timed(self.inner.chat_completion_stream(request)).await {
                Ok(stream) => return Ok(idle_bounded(stream, self.stream_idle)),
                Err(e) if self.should_retry(&e) => {
                    let wait = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    sleep(wait).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn anthropic_messages(
        &self,
        request: &anthropic::Request,
    ) -> Result<anthropic::Response, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self.timed(self.inner.anthropic_messages(request)).await {
                Ok(resp) => return Ok(resp),
                Err(e) if self.should_retry(&e) => {
                    let wait = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    sleep(wait).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &anthropic::Request,
    ) -> Result<BoxStream<'static, Result<anthropic::StreamEvent, Error>>, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self
                .timed(self.inner.anthropic_messages_stream(request))
                .await
            {
                Ok(stream) => return Ok(idle_bounded(stream, self.stream_idle)),
                Err(e) if self.should_retry(&e) => {
                    let wait = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    sleep(wait).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &gemini::Request,
    ) -> Result<BoxStream<'static, Result<gemini::Response, Error>>, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self
                .timed(self.inner.gemini_generate_content_stream(model, request))
                .await
            {
                Ok(stream) => return Ok(idle_bounded(stream, self.stream_idle)),
                Err(e) if self.should_retry(&e) => {
                    let wait = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    sleep(wait).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        self.inner.embedding(request).await
    }

    /// Passed through unretried — an idempotent listing that callers make
    /// once, and a failure here is the answer, not something to sit through.
    async fn models(&self) -> Result<crate::ModelList, Error> {
        self.inner.models().await
    }

    async fn image_generation(
        &self,
        request: &ImageRequest,
    ) -> Result<(bytes::Bytes, String), Error> {
        self.inner.image_generation(request).await
    }

    async fn audio_speech(
        &self,
        request: &AudioSpeechRequest,
    ) -> Result<(bytes::Bytes, String), Error> {
        self.inner.audio_speech(request).await
    }

    async fn audio_transcription(
        &self,
        model: &str,
        fields: &[MultipartField],
    ) -> Result<(bytes::Bytes, String), Error> {
        self.inner.audio_transcription(model, fields).await
    }
}

/// Fail a stream that goes `idle` without producing a chunk.
///
/// The timeout is per-chunk, not per-stream: a long generation is fine, a
/// silent one is not. Terminates the stream on expiry — a stalled upstream
/// has no more to say, and the events already yielded stay valid.
fn idle_bounded<T: Send + 'static>(
    stream: BoxStream<'static, Result<T, Error>>,
    idle: Duration,
) -> BoxStream<'static, Result<T, Error>> {
    if idle.is_zero() {
        return stream;
    }
    Box::pin(futures_util::stream::unfold(
        Some(stream),
        move |state| async move {
            let mut stream = state?;
            match timeout(idle, stream.next()).await {
                Ok(Some(item)) => Some((item, Some(stream))),
                Ok(None) => None,
                Err(_) => Some((Err(Error::Timeout), None)),
            }
        },
    ))
}

/// Full jitter: random duration in [backoff/2, backoff].
fn jittered(backoff: Duration) -> Duration {
    let lo = backoff.as_millis() as u64 / 2;
    let hi = backoff.as_millis() as u64;
    if lo >= hi {
        return backoff;
    }
    Duration::from_millis(random_millis(lo, hi))
}

/// Uniform random in `[lo, hi]`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn random_millis(lo: u64, hi: u64) -> u64 {
    use rand::Rng;
    rand::rng().random_range(lo..=hi)
}

/// `Math.random` counterpart of `rand`'s `random_range`. Jitter only needs to
/// spread retries across callers, so the browser's non-cryptographic PRNG is
/// the right tool and saves pulling `getrandom` into the build.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn random_millis(lo: u64, hi: u64) -> u64 {
    lo + (js_sys::Math::random() * (hi - lo + 1) as f64) as u64
}

/// wasi:clocks counterpart of `tokio::time::sleep`.
#[cfg(target_os = "wasi")]
async fn sleep(duration: Duration) {
    wstd::task::sleep(duration.into()).await;
}

/// wasi:clocks counterpart of `tokio::time::timeout`: `Err` once `duration`
/// elapses.
#[cfg(target_os = "wasi")]
async fn timeout<F: Future>(duration: Duration, fut: F) -> std::io::Result<F::Output> {
    wstd::future::FutureExt::timeout(fut, wstd::time::Duration::from(duration)).await
}

/// `setTimeout` counterpart of `tokio::time::sleep`.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
async fn sleep(duration: Duration) {
    let millis = duration.as_millis().try_into().unwrap_or(u32::MAX);
    crate::SingleThreaded(gloo_timers::future::TimeoutFuture::new(millis)).await;
}

/// `setTimeout` counterpart of `tokio::time::timeout`: `Err` once `duration`
/// elapses.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
async fn timeout<F: Future>(duration: Duration, fut: F) -> Result<F::Output, ()> {
    use futures_util::future::{Either, select};
    match select(std::pin::pin!(fut), std::pin::pin!(sleep(duration))).await {
        Either::Left((output, _)) => Ok(output),
        Either::Right(_) => Err(()),
    }
}

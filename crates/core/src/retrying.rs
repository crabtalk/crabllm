use crate::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, AudioSpeechRequest, BoxStream,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, Error, GeminiRequest, GeminiResponse, ImageRequest, MultipartField,
    Provider,
};
use rand::Rng;
use std::{future::Future, time::Duration};

const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
const INITIAL_BACKOFF: Duration = Duration::from_millis(100);

/// A `Provider` wrapper that retries transient failures with exponential
/// backoff and full jitter, and bounds each attempt with a per-call timeout.
///
/// For streaming methods the per-attempt timeout covers only opening the
/// stream; a mid-stream stall is bounded by the transport's read timeout.
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
}

impl<P: Provider> Retrying<P> {
    /// Wrap a provider with the default retry policy
    /// (2 retries, 30s timeout, 60s max retry-after, 100ms initial backoff).
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            max_retries: DEFAULT_MAX_RETRIES,
            timeout: DEFAULT_TIMEOUT,
            max_retry_after: DEFAULT_MAX_RETRY_AFTER,
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
        let Ok(result) = tokio::time::timeout(self.timeout, fut).await else {
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
                    let sleep = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    tokio::time::sleep(sleep).await;
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
                Ok(stream) => return Ok(stream),
                Err(e) if self.should_retry(&e) => {
                    let sleep = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    tokio::time::sleep(sleep).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self.timed(self.inner.anthropic_messages(request)).await {
                Ok(resp) => return Ok(resp),
                Err(e) if self.should_retry(&e) => {
                    let sleep = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    tokio::time::sleep(sleep).await;
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("retry loop exited without producing an error"))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self
                .timed(self.inner.anthropic_messages_stream(request))
                .await
            {
                Ok(stream) => return Ok(stream),
                Err(e) if self.should_retry(&e) => {
                    let sleep = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    tokio::time::sleep(sleep).await;
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
        request: &GeminiRequest,
    ) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
        let mut backoff = INITIAL_BACKOFF;
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self
                .timed(self.inner.gemini_generate_content_stream(model, request))
                .await
            {
                Ok(stream) => return Ok(stream),
                Err(e) if self.should_retry(&e) => {
                    let sleep = e.retry_after().unwrap_or_else(|| jittered(backoff));
                    last_err = Some(e);
                    tokio::time::sleep(sleep).await;
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

/// Full jitter: random duration in [backoff/2, backoff].
fn jittered(backoff: Duration) -> Duration {
    let lo = backoff.as_millis() as u64 / 2;
    let hi = backoff.as_millis() as u64;
    if lo >= hi {
        return backoff;
    }
    Duration::from_millis(rand::rng().random_range(lo..=hi))
}

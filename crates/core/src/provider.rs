use crate::{
    AudioSpeechRequest, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    EmbeddingRequest, EmbeddingResponse, Error, ImageRequest, MultipartField,
};
use bytes::Bytes;
use futures_core::Stream;
use std::{future::Future, pin::Pin};

/// A boxed, `Send`, dynamically-typed stream — used by `Provider` so the
/// trait can return a uniform stream type without each implementor leaking
/// its concrete combinator chain through an associated type.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// The dispatch surface every provider implementation satisfies.
///
/// Returns futures via RPITIT (return-position `impl Trait` in trait) so the
/// type system can monomorphize through the trait without dyn dispatch or
/// per-call boxing. The proxy crate is generic over `P: Provider`; the binary
/// crate picks the concrete type by defining a workspace-level union enum
/// that delegates each method.
///
/// Streaming responses use `BoxStream` because returning an opaque stream
/// from an async-returning trait method requires a fixed type at the trait
/// boundary; the boxing is one allocation per stream creation, not per item.
/// **Implementors must clone any borrowed data from the request before
/// returning the stream** — the returned `BoxStream` is `'static` and cannot
/// borrow from the request reference.
///
/// The optional methods (`embedding`, `image_generation`, `audio_speech`,
/// `audio_transcription`) default to returning `Error::not_implemented`, so
/// concrete providers only override the methods they actually support.
/// Overrides are free to capture `self` or the request reference — only the
/// default impl bodies happen to capture nothing, and that's an
/// implementation detail of the defaults, not a constraint on the trait.
pub trait Provider: Send + Sync {
    fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> impl Future<Output = Result<ChatCompletionResponse, Error>> + Send;

    fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> impl Future<Output = Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error>> + Send;

    fn embedding(
        &self,
        _request: &EmbeddingRequest,
    ) -> impl Future<Output = Result<EmbeddingResponse, Error>> + Send {
        async { Err(Error::not_implemented("embedding")) }
    }

    fn image_generation(
        &self,
        _request: &ImageRequest,
    ) -> impl Future<Output = Result<(Bytes, String), Error>> + Send {
        async { Err(Error::not_implemented("image_generation")) }
    }

    fn audio_speech(
        &self,
        _request: &AudioSpeechRequest,
    ) -> impl Future<Output = Result<(Bytes, String), Error>> + Send {
        async { Err(Error::not_implemented("audio_speech")) }
    }

    fn audio_transcription(
        &self,
        _model: &str,
        _fields: &[MultipartField],
    ) -> impl Future<Output = Result<(Bytes, String), Error>> + Send {
        async { Err(Error::not_implemented("audio_transcription")) }
    }

    /// Whether this provider speaks the OpenAI wire format and can forward
    /// raw JSON bytes without deserialization.
    fn is_openai_compat(&self) -> bool {
        false
    }

    /// Whether this provider speaks the Anthropic wire format and can
    /// forward raw `/v1/messages` bytes without translation.
    fn is_anthropic_compat(&self) -> bool {
        false
    }

    /// Forward raw OpenAI-format JSON body and return raw response bytes.
    /// The default deserializes, calls [`chat_completion`](Self::chat_completion),
    /// and re-serializes. OpenAI-compatible providers override to skip serde.
    fn chat_completion_raw(
        &self,
        _model: &str,
        raw_body: Bytes,
    ) -> impl Future<Output = Result<Bytes, Error>> + Send {
        async move {
            let request: ChatCompletionRequest = crate::json::from_slice(&raw_body)
                .map_err(|e| Error::Internal(e.to_string()))?;
            let resp = self.chat_completion(&request).await?;
            Ok(Bytes::from(
                crate::json::to_vec(&resp).map_err(|e| Error::Internal(e.to_string()))?,
            ))
        }
    }

    /// Forward raw Anthropic-format JSON body and return raw response bytes.
    /// The default translates Anthropic → OpenAI, calls [`chat_completion`],
    /// and translates back. The Anthropic provider overrides to skip serde.
    fn anthropic_messages_raw(
        &self,
        _raw_body: Bytes,
    ) -> impl Future<Output = Result<Bytes, Error>> + Send {
        async { Err(Error::not_implemented("anthropic_messages_raw")) }
    }
}

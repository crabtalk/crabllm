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
/// Default implementations return `Error::not_implemented` so concrete
/// providers only need to override the methods they actually support. The
/// default impls intentionally capture neither `self` nor the request,
/// keeping the returned future `'static` and `Send` regardless of
/// implementor.
pub trait Provider: Send + Sync {
    fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> impl Future<Output = Result<ChatCompletionResponse, Error>> + Send;

    fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> impl Future<
        Output = Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error>,
    > + Send;

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
}

use crate::client::{ANTHROPIC_VERSION, Client, RawClient, Route, route};
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error,
    GeminiRequest, GeminiResponse, ModelList, Provider, codec,
};
use futures::StreamExt;

const ANTHROPIC: &[(&str, &str)] = &[("anthropic-version", ANTHROPIC_VERSION)];

/// The actual HTTP dispatch. Streaming responses are parsed by the shared
/// `crabllm_core::codec`, so the client and gateway can never drift.
impl Provider for RawClient {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let bytes = self
            .post_checked("/v1/chat/completions", &[], body.into())
            .await?;
        crabllm_core::json::from_slice(&bytes).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Encode(e.to_string()))?;
        let byte_stream = self
            .post_sse("/v1/chat/completions", &[], body.into())
            .await?;
        Ok(codec::openai::sse_stream(byte_stream).boxed())
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let bytes = self
            .post_checked("/v1/messages", ANTHROPIC, body.into())
            .await?;
        crabllm_core::json::from_slice(&bytes).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        let mut req = request.clone();
        req.stream = Some(true);
        let model = req.model.clone();
        let body = crabllm_core::json::to_vec(&req).map_err(|e| Error::Encode(e.to_string()))?;
        let byte_stream = self
            .post_sse("/v1/messages", ANTHROPIC, body.into())
            .await?;
        Ok(codec::anthropic::anthropic_event_stream(byte_stream, model).boxed())
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let bytes = self
            .post_checked("/v1/embeddings", &[], body.into())
            .await?;
        crabllm_core::json::from_slice(&bytes).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn models(&self) -> Result<ModelList, Error> {
        self.get_models().await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let path = format!("/v1beta/models/{model}:streamGenerateContent?alt=sse");
        let byte_stream = self.post_sse(&path, &[], body.into()).await?;
        Ok(codec::gemini::gemini_event_stream(byte_stream).boxed())
    }
}

/// `Client` delegates to its inner `Retrying<RawClient>`, which adds retries and
/// the per-attempt timeout before handing off to the dispatch above.
impl Provider for Client {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        self.inner.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        self.inner.chat_completion_stream(request).await
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        if !self.bridge {
            return self.inner.anthropic_messages(request).await;
        }
        match route(&self.model_dialects(&request.model).await) {
            Route::Native => self.inner.anthropic_messages(request).await,
            Route::Translate => self.translate_anthropic(request).await,
            // Unknown model: try native, and on *any* native failure fall back
            // to translation. We deliberately don't distinguish a dialect miss
            // from other errors (that means sniffing the error body), so we log
            // the native error rather than drop it before translating.
            Route::NativeElseTranslate => match self.inner.anthropic_messages(request).await {
                Ok(resp) => Ok(resp),
                Err(native_err) => {
                    tracing::debug!(
                        model = %request.model,
                        "native /v1/messages failed for unknown model, translating: {native_err}"
                    );
                    self.translate_anthropic(request).await
                }
            },
        }
    }

    async fn anthropic_messages_stream(
        &self,
        request: &AnthropicRequest,
    ) -> Result<BoxStream<'static, Result<AnthropicStreamEvent, Error>>, Error> {
        if !self.bridge {
            return self.inner.anthropic_messages_stream(request).await;
        }
        // Parallel to `anthropic_messages` above — keep the two in sync.
        match route(&self.model_dialects(&request.model).await) {
            Route::Native => self.inner.anthropic_messages_stream(request).await,
            Route::Translate => self.translate_anthropic_stream(request).await,
            Route::NativeElseTranslate => {
                match self.inner.anthropic_messages_stream(request).await {
                    Ok(stream) => Ok(stream),
                    Err(native_err) => {
                        tracing::debug!(
                            model = %request.model,
                            "native /v1/messages failed for unknown model, translating: {native_err}"
                        );
                        self.translate_anthropic_stream(request).await
                    }
                }
            }
        }
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        self.inner.embedding(request).await
    }

    async fn models(&self) -> Result<ModelList, Error> {
        self.inner.models().await
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
        self.inner
            .gemini_generate_content_stream(model, request)
            .await
    }
}

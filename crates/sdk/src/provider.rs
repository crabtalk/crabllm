use crate::client::{ANTHROPIC_VERSION, Client};
use crabllm_core::{
    AnthropicRequest, AnthropicResponse, AnthropicStreamEvent, BoxStream, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse, Error,
    GeminiRequest, GeminiResponse, Provider, codec,
};
use futures::StreamExt;

const JSON: &str = "application/json";

impl Provider for Client {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let auth = self.bearer();
        let headers = [("content-type", JSON), ("authorization", auth.as_str())];
        let bytes = self
            .http
            .post(&self.url("/v1/chat/completions"), &headers, body.into())
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
        let auth = self.bearer();
        let headers = [("content-type", JSON), ("authorization", auth.as_str())];
        let byte_stream = self
            .http
            .post_stream(&self.url("/v1/chat/completions"), &headers, body.into())
            .await?;
        Ok(codec::openai::sse_stream(byte_stream).boxed())
    }

    async fn anthropic_messages(
        &self,
        request: &AnthropicRequest,
    ) -> Result<AnthropicResponse, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let auth = self.bearer();
        let headers = [
            ("content-type", JSON),
            ("authorization", auth.as_str()),
            ("anthropic-version", ANTHROPIC_VERSION),
        ];
        let bytes = self
            .http
            .post(&self.url("/v1/messages"), &headers, body.into())
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
        let auth = self.bearer();
        let headers = [
            ("content-type", JSON),
            ("authorization", auth.as_str()),
            ("anthropic-version", ANTHROPIC_VERSION),
        ];
        let byte_stream = self
            .http
            .post_stream(&self.url("/v1/messages"), &headers, body.into())
            .await?;
        Ok(codec::anthropic::anthropic_event_stream(byte_stream, model).boxed())
    }

    async fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let auth = self.bearer();
        let headers = [("content-type", JSON), ("authorization", auth.as_str())];
        let bytes = self
            .http
            .post(&self.url("/v1/embeddings"), &headers, body.into())
            .await?;
        crabllm_core::json::from_slice(&bytes).map_err(|e| Error::Decode(e.to_string()))
    }

    async fn gemini_generate_content_stream(
        &self,
        model: &str,
        request: &GeminiRequest,
    ) -> Result<BoxStream<'static, Result<GeminiResponse, Error>>, Error> {
        let body = crabllm_core::json::to_vec(request).map_err(|e| Error::Encode(e.to_string()))?;
        let auth = self.bearer();
        let headers = [("content-type", JSON), ("authorization", auth.as_str())];
        let path = format!("/v1beta/models/{model}:streamGenerateContent?alt=sse");
        let byte_stream = self
            .http
            .post_stream(&self.url(&path), &headers, body.into())
            .await?;
        Ok(codec::gemini::gemini_event_stream(byte_stream).boxed())
    }
}

//! `MlxProvider` — `Provider`-trait front door backed by an [`MlxPool`].
//!
//! Each trait call routes by `request.model`, loads the matching
//! [`MlxModel`] on demand via [`MlxPool::ensure_loaded`], and delegates
//! to the model's `Provider` implementation. Cloning is cheap:
//! `Arc<MlxPool>` under the hood.

use crate::{model::MlxModel, pool::MlxPool};
use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Error, Provider,
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MlxProvider {
    pool: Arc<MlxPool>,
}

impl MlxProvider {
    pub fn new(pool: Arc<MlxPool>) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &Arc<MlxPool> {
        &self.pool
    }

    async fn resolve(&self, model_id: &str) -> Result<MlxModel, Error> {
        self.pool.ensure_loaded(model_id).await
    }
}

impl Provider for MlxProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        let model = self.resolve(&request.model).await?;
        model.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        let model = self.resolve(&request.model).await?;
        model.chat_completion_stream(request).await
    }

    // embedding / image / audio fall through to the trait defaults
    // (`Error::not_implemented`). Those modalities land in their own
    // follow-up PR sets — see CONTRIBUTING / plan docs.
}

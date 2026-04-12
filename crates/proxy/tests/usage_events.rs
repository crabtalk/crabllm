//! Integration test for the [`UsageEvent`] broadcast channel.
//!
//! Drives a real request through the proxy router with a fake
//! provider and storage, subscribes to `usage_events`, and asserts
//! that exactly one event arrives with the expected shape.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use crabllm_core::{
    BoxFuture, BoxStream, ChatCompletionRequest, ChatCompletionResponse, Choice, Error,
    FinishReason, GatewayConfig, KvPairs, Message, Prefix, Provider, Role, Storage, Usage,
};
use crabllm_provider::{Deployment, ProviderRegistry};
use crabllm_proxy::{AppState, UsageEvent, router};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::sync::broadcast;
use tower::ServiceExt;

struct FakeProvider;

impl Provider for FakeProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        Ok(ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: request.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: Some(serde_json::Value::String("hi".into())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                    extra: Default::default(),
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            usage: Some(Usage {
                prompt_tokens: 11,
                completion_tokens: 22,
                total_tokens: 33,
                completion_tokens_details: None,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
            }),
            system_fingerprint: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<crabllm_core::ChatCompletionChunk, Error>>, Error> {
        Err(Error::not_implemented("stream"))
    }
}

struct FakeStorage;

impl Storage for FakeStorage {
    fn get(&self, _key: &[u8]) -> BoxFuture<'_, Result<Option<Vec<u8>>, Error>> {
        Box::pin(async { Ok(None) })
    }
    fn set(&self, _key: &[u8], _value: Vec<u8>) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }
    fn increment(&self, _key: &[u8], _delta: i64) -> BoxFuture<'_, Result<i64, Error>> {
        Box::pin(async { Ok(0) })
    }
    fn list(&self, _prefix: &Prefix) -> BoxFuture<'_, Result<KvPairs, Error>> {
        Box::pin(async { Ok(KvPairs::default()) })
    }
    fn delete(&self, _key: &[u8]) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }
}

fn empty_config() -> GatewayConfig {
    GatewayConfig {
        listen: String::new(),
        providers: HashMap::new(),
        keys: Vec::new(),
        extensions: None,
        storage: None,
        aliases: HashMap::new(),
        pricing: HashMap::new(),
        admin_token: None,
        shutdown_timeout: 30,
    }
}

fn build_state(tx: broadcast::Sender<UsageEvent>) -> AppState<FakeStorage, FakeProvider> {
    let mut providers: HashMap<String, Vec<Arc<Deployment<FakeProvider>>>> = HashMap::new();
    providers.insert(
        "fake-model".to_string(),
        vec![Arc::new(Deployment {
            provider: FakeProvider,
            weight: 1,
            max_retries: 0,
            timeout: Duration::from_secs(5),
        })],
    );
    let mut model_providers = HashMap::new();
    model_providers.insert("fake-model".to_string(), "fake".to_string());

    let registry = ProviderRegistry::new(providers, HashMap::new(), model_providers);

    AppState {
        registry,
        config: empty_config(),
        extensions: Arc::new(Vec::new()),
        storage: Arc::new(FakeStorage),
        key_map: Arc::new(RwLock::new(HashMap::new())),
        usage_events: Some(tx),
    }
}

#[tokio::test]
async fn chat_completion_emits_one_usage_event() {
    let (tx, mut rx) = broadcast::channel::<UsageEvent>(16);
    let state = build_state(tx);
    let app = router(state, vec![]);

    let body = serde_json::json!({
        "model": "fake-model",
        "messages": [{ "role": "user", "content": "hi" }],
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let event = tokio::time::timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("timed out waiting for UsageEvent")
        .expect("broadcast closed");

    assert_eq!(event.endpoint, "chat.completions");
    assert_eq!(event.model, "fake-model");
    assert_eq!(event.provider, "fake");
    assert_eq!(event.tokens_in, 11);
    assert_eq!(event.tokens_out, 22);
    assert_eq!(event.status, 200);
    assert!(event.error.is_none());

    // No duplicate emission. Either a timeout (channel still open but
    // silent) or a Closed error (all senders dropped after the handler
    // returned) is acceptable — both mean "no further events".
    match tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {
        Err(_) => {}                                          // timeout
        Ok(Err(broadcast::error::RecvError::Closed)) => {}    // channel closed
        Ok(Err(broadcast::error::RecvError::Lagged(_))) => {} // treat as no extra
        Ok(Ok(evt)) => panic!("unexpected second event: {:?}", evt),
    }
}

#[tokio::test]
async fn none_usage_events_is_zero_cost() {
    let mut providers: HashMap<String, Vec<Arc<Deployment<FakeProvider>>>> = HashMap::new();
    providers.insert(
        "fake-model".to_string(),
        vec![Arc::new(Deployment {
            provider: FakeProvider,
            weight: 1,
            max_retries: 0,
            timeout: Duration::from_secs(5),
        })],
    );
    let mut model_providers = HashMap::new();
    model_providers.insert("fake-model".to_string(), "fake".to_string());
    let registry = ProviderRegistry::new(providers, HashMap::new(), model_providers);

    let state = AppState::<FakeStorage, FakeProvider> {
        registry,
        config: empty_config(),
        extensions: Arc::new(Vec::new()),
        storage: Arc::new(FakeStorage),
        key_map: Arc::new(RwLock::new(HashMap::new())),
        usage_events: None,
    };

    let app = router(state, vec![]);
    let body = serde_json::json!({
        "model": "fake-model",
        "messages": [{ "role": "user", "content": "hi" }],
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

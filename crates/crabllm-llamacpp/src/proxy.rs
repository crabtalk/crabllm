use crate::pool::ServerPool;
use axum::{
    Router,
    body::Body,
    extract::{OriginalUri, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
};
use std::sync::Arc;

/// Shared state for the proxy handlers.
#[derive(Clone)]
pub struct ProxyState {
    pub pool: Arc<ServerPool>,
    pub client: reqwest::Client,
    pub models: Vec<String>,
}

/// Build the router for the model-routing proxy.
pub fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        // Catch-all for POST routes — forward to the right llama-server.
        .fallback(proxy_request)
        .with_state(state)
}

/// Proxy a request to the correct llama-server based on the `model` field.
async fn proxy_request(
    State(state): State<ProxyState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let model = match extract_model(&body) {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "missing 'model' field in request body"})),
            )
                .into_response();
        }
    };

    let base_url = match state.pool.ensure_running(&model).await {
        Ok(url) => url,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    // base_url is "http://127.0.0.1:{port}/v1", uri.path() is "/v1/chat/completions".
    // Strip the /v1 prefix from the uri and append to base_url.
    let path = uri.path().strip_prefix("/v1").unwrap_or(uri.path());
    let url = format!("{base_url}{path}");

    forward(&state.client, &url, &headers, body).await
}

/// Forward a request and stream the response back.
async fn forward(
    client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let mut req = client.post(url).body(body.to_vec());
    if let Some(ct) = headers.get("content-type") {
        req = req.header("content-type", ct);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("upstream error: {e}")})),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let resp_headers = resp.headers().clone();
    let stream = resp.bytes_stream();

    let mut builder = Response::builder().status(status);
    for (key, value) in &resp_headers {
        builder = builder.header(key, value);
    }
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Extract the `model` field from a JSON body.
fn extract_model(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v["model"].as_str().map(|s| s.to_string())
}

/// List all configured models.
async fn list_models(State(state): State<ProxyState>) -> Json<serde_json::Value> {
    let data: Vec<serde_json::Value> = state
        .models
        .iter()
        .map(|name| {
            serde_json::json!({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "crabllm-llamacpp",
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
}

/// Health check.
async fn health() -> StatusCode {
    StatusCode::OK
}

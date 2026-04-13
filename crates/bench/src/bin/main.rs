use axum::{
    Router,
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use crabllm_core::{
    ApiError, ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, Delta, Embedding,
    EmbeddingResponse, EmbeddingUsage, FinishReason, Message, Model, ModelList, Role, Usage,
};
use futures::{StreamExt, stream};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "crabllm-bench", about = "Mock OpenAI backend for benchmarking")]
struct Cli {
    /// Port to listen on
    #[arg(short, long, default_value = "9999")]
    port: u16,

    /// Number of SSE chunks for streaming responses
    #[arg(short, long, default_value = "20")]
    chunks: usize,

    /// Delay in milliseconds between streaming chunks (or before non-streaming response)
    #[arg(short, long, default_value = "0")]
    delay: u64,

    /// Fraction of requests that return HTTP 500 (0.0 to 1.0)
    #[arg(short, long, default_value = "0.0")]
    error_rate: f64,
}

/// Pre-serialized canned responses, computed once at startup.
struct Canned {
    chat_json: String,
    embed_json: String,
    models_json: String,
    error_json: String,
    chunks: usize,
    delay_ms: u64,
    error_rate: f64,
}

impl Canned {
    fn should_error(&self) -> bool {
        self.error_rate > 0.0 && rand::random::<f64>() < self.error_rate
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let error_body = ApiError::new("simulated upstream failure", "server_error");
    let canned = Arc::new(Canned {
        chat_json: serde_json::to_string(&canned_chat_response()).unwrap(),
        embed_json: serde_json::to_string(&canned_embed_response()).unwrap(),
        models_json: serde_json::to_string(&canned_models()).unwrap(),
        error_json: serde_json::to_string(&error_body).unwrap(),
        chunks: cli.chunks,
        delay_ms: cli.delay,
        error_rate: cli.error_rate,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .with_state(canned);

    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!(
        "mock backend listening on {addr} (chunks={}, delay={}ms, error_rate={})",
        cli.chunks, cli.delay, cli.error_rate
    );
    axum::serve(NoDelayListener(listener), app).await.unwrap();
}

async fn chat_completions(
    axum::extract::State(canned): axum::extract::State<Arc<Canned>>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    if canned.should_error() {
        return (StatusCode::INTERNAL_SERVER_ERROR, canned.error_json.clone()).into_response();
    }

    let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("stream")?.as_bool())
        .unwrap_or(false);

    if is_stream {
        let n = canned.chunks;
        let delay_ms = canned.delay_ms;
        let chunks = stream::unfold(0usize, move |i| async move {
            if i >= n {
                return None;
            }
            if delay_ms > 0 && i > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            let is_last = i == n - 1;
            let chunk = ChatCompletionChunk {
                id: "chatcmpl-bench".into(),
                object: "chat.completion.chunk".into(),
                created: 0,
                model: "bench-chat".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: if i == 0 { Some(Role::Assistant) } else { None },
                        content: if is_last { None } else { Some("word ".into()) },
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    finish_reason: if is_last {
                        Some(FinishReason::Stop)
                    } else {
                        None
                    },
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
            };
            let json = serde_json::to_string(&chunk).unwrap();
            let event = Ok::<_, std::convert::Infallible>(Event::default().data(json));
            Some((event, i + 1))
        });

        let done = stream::once(async {
            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
        });
        Sse::new(chunks.chain(done)).into_response()
    } else {
        if canned.delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(canned.delay_ms)).await;
        }
        (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            canned.chat_json.clone(),
        )
            .into_response()
    }
}

async fn embeddings(
    axum::extract::State(canned): axum::extract::State<Arc<Canned>>,
) -> impl IntoResponse {
    if canned.should_error() {
        return (StatusCode::INTERNAL_SERVER_ERROR, canned.error_json.clone()).into_response();
    }
    if canned.delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(canned.delay_ms)).await;
    }
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        canned.embed_json.clone(),
    )
        .into_response()
}

async fn models(
    axum::extract::State(canned): axum::extract::State<Arc<Canned>>,
) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        canned.models_json.clone(),
    )
}

fn canned_chat_response() -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: "chatcmpl-bench".into(),
        object: "chat.completion".into(),
        created: 0,
        model: "bench-chat".into(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: Some(serde_json::Value::String(
                    "This is a benchmark response.".into(),
                )),
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
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            completion_tokens_details: None,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
        }),
        system_fingerprint: None,
    }
}

fn canned_embed_response() -> EmbeddingResponse {
    EmbeddingResponse {
        object: "list".into(),
        data: vec![Embedding {
            object: "embedding".into(),
            index: 0,
            embedding: vec![0.0; 1536],
        }],
        model: "bench-embed".into(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    }
}

struct NoDelayListener(tokio::net::TcpListener);

impl axum::serve::Listener for NoDelayListener {
    type Io = tokio::net::TcpStream;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.0.accept().await {
                Ok((stream, addr)) => {
                    let _ = stream.set_nodelay(true);
                    return (stream, addr);
                }
                Err(_) => continue,
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.0.local_addr()
    }
}

fn canned_models() -> ModelList {
    ModelList {
        object: "list".into(),
        data: vec![
            Model {
                id: "bench-chat".into(),
                object: "model".into(),
                created: 0,
                owned_by: "mock".into(),
                context_length: None,
                pricing: None,
                vision: None,
            },
            Model {
                id: "bench-embed".into(),
                object: "model".into(),
                created: 0,
                owned_by: "mock".into(),
                context_length: None,
                pricing: None,
                vision: None,
            },
        ],
    }
}

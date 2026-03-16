use axum::{
    Router,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, Delta, Embedding,
    EmbeddingResponse, EmbeddingUsage, Message, Model, ModelList, Usage,
};
use futures::stream;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "crabtalk-bench",
    about = "Mock OpenAI backend for benchmarking"
)]
struct Cli {
    /// Port to listen on
    #[arg(short, long, default_value = "9999")]
    port: u16,

    /// Number of SSE chunks for streaming responses
    #[arg(short, long, default_value = "20")]
    chunks: usize,
}

/// Pre-serialized canned responses, computed once at startup.
struct Canned {
    chat_json: String,
    embed_json: String,
    models_json: String,
    chunks: usize,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let canned = Arc::new(Canned {
        chat_json: serde_json::to_string(&canned_chat_response()).unwrap(),
        embed_json: serde_json::to_string(&canned_embed_response()).unwrap(),
        models_json: serde_json::to_string(&canned_models()).unwrap(),
        chunks: cli.chunks,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .with_state(canned);

    let addr = format!("127.0.0.1:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("mock backend listening on {addr} (chunks={})", cli.chunks);
    axum::serve(listener, app).await.unwrap();
}

async fn chat_completions(
    axum::extract::State(canned): axum::extract::State<Arc<Canned>>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    // Check if streaming is requested by looking for "stream":true in the body.
    let is_stream = body
        .windows(13)
        .any(|w| w == b"\"stream\":true" || w == b"\"stream\": true");

    if is_stream {
        let n = canned.chunks;
        let chunks = (0..n).map(move |i| {
            let is_last = i == n - 1;
            let chunk = ChatCompletionChunk {
                id: "chatcmpl-bench".into(),
                object: "chat.completion.chunk".into(),
                created: 0,
                model: "bench-chat".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: if i == 0 {
                            Some("assistant".into())
                        } else {
                            None
                        },
                        content: if is_last { None } else { Some("word ".into()) },
                        tool_calls: None,
                    },
                    finish_reason: if is_last { Some("stop".into()) } else { None },
                }],
                usage: None,
            };
            let json = serde_json::to_string(&chunk).unwrap();
            Ok::<_, std::convert::Infallible>(Event::default().data(json))
        });

        let done = std::iter::once(Ok(Event::default().data("[DONE]")));
        Sse::new(stream::iter(chunks.chain(done))).into_response()
    } else {
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
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        canned.embed_json.clone(),
    )
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
                role: "assistant".into(),
                content: Some(serde_json::Value::String(
                    "This is a benchmark response.".into(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("stop".into()),
        }],
        usage: Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        }),
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

fn canned_models() -> ModelList {
    ModelList {
        object: "list".into(),
        data: vec![
            Model {
                id: "bench-chat".into(),
                object: "model".into(),
                created: 0,
                owned_by: "mock".into(),
            },
            Model {
                id: "bench-embed".into(),
                object: "model".into(),
                created: 0,
                owned_by: "mock".into(),
            },
        ],
    }
}

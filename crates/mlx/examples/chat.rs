//! Interactive chat with a local MLX model.
//!
//! Usage:
//!   cargo run -p crabllm-mlx --example chat -- mlx-community/Qwen3.5-2B-MLX-4bit
//!   cargo run -p crabllm-mlx --example chat -- /path/to/cached/model
//!
//! The first run downloads the model from HuggingFace (may take a few
//! minutes depending on model size and network speed). Subsequent runs
//! load from cache instantly.
//!
//! Type a message and press Enter. The model streams its response
//! token by token. Type `exit` or Ctrl-C to quit.

use clap::Parser;
use crabllm_core::{ChatCompletionRequest, Message, Provider};
use crabllm_mlx::{MlxPool, MlxProvider};
use futures::StreamExt;
use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};

#[derive(Parser)]
#[command(name = "chat", about = "Interactive MLX chat")]
struct Cli {
    /// Model name: HuggingFace repo id or local directory path.
    model: String,

    /// Idle timeout in seconds for the model pool (default: 1800).
    #[arg(long, default_value = "1800")]
    idle_timeout: u64,
}

#[tokio::main]
async fn main() {
    // Default to info-level logging so download progress is visible.
    if std::env::var("RUST_LOG").is_err() {
        // SAFETY: called before any threads are spawned.
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let pool = match MlxPool::new(cli.idle_timeout) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("error: failed to create pool: {e}");
            std::process::exit(1);
        }
    };
    let provider = MlxProvider::new(pool);
    let mut history: Vec<Message> = Vec::new();

    eprintln!("model: {}", cli.model);
    eprintln!("loading model (first run downloads from HuggingFace)...");

    // Warm up: trigger the download + model load before the REPL so
    // the user sees progress instead of a blank hang on first message.
    let warmup = ChatCompletionRequest {
        model: cli.model.clone(),
        messages: vec![Message::user("hi")],
        temperature: None,
        top_p: None,
        max_tokens: Some(1),
        stream: None,
        stop: None,
        tools: None,
        tool_choice: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        user: None,
        reasoning_effort: None,
        extra: serde_json::Map::new(),
    };
    match provider.chat_completion(&warmup).await {
        Ok(_) => eprintln!("model loaded.\n"),
        Err(e) => {
            eprintln!("error: failed to load model: {e}");
            std::process::exit(1);
        }
    }

    eprintln!("type a message, press Enter to send. \"exit\" to quit.\n");

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        history.push(Message::user(line));

        let request = ChatCompletionRequest {
            model: cli.model.clone(),
            messages: history.clone(),
            temperature: Some(0.7),
            top_p: None,
            max_tokens: None,
            stream: Some(true),
            stop: None,
            tools: None,
            tool_choice: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            user: None,
            reasoning_effort: None,
            extra: serde_json::Map::new(),
        };

        match provider.chat_completion_stream(&request).await {
            Ok(mut stream) => {
                let mut full_response = String::new();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            if let Some(text) = c.content() {
                                print!("{text}");
                                io::stdout().flush().ok();
                                full_response.push_str(text);
                            }
                        }
                        Err(e) => {
                            eprintln!("\nerror: {e}");
                            break;
                        }
                    }
                }
                println!();
                if !full_response.is_empty() {
                    history.push(Message::assistant(&full_response));
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
            }
        }
    }
}

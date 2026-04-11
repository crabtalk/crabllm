//! Interactive chat with a local MLX model.
//!
//! Usage:
//!   cargo run -p crabllm-mlx --example chat -- qwen3-4b-4bit
//!   cargo run -p crabllm-mlx --example chat -- mlx-community/Qwen3.5-2B-MLX-4bit
//!   cargo run -p crabllm-mlx --example chat -- --list
//!
//! Downloads the model on first run (with progress), then loads from
//! cache. Type a message and press Enter. `exit` or Ctrl-C to quit.

use clap::Parser;
use crabllm_core::{ChatCompletionRequest, Message, Provider};
use crabllm_mlx::{DownloadEvent, MlxPool, MlxProvider};
use futures::StreamExt;
use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};

#[derive(Parser)]
#[command(name = "chat", about = "Interactive MLX chat")]
struct Cli {
    /// Model name: alias, HuggingFace repo id, or local directory path.
    model: Option<String>,

    /// List all predefined models and exit.
    #[arg(long)]
    list: bool,

    /// Idle timeout in seconds for the model pool (default: 1800).
    #[arg(long, default_value = "1800")]
    idle_timeout: u64,
}

/// Ensure the model is downloaded, showing progress on stderr.
fn ensure_downloaded(model: &str) {
    // Resolve alias to repo id.
    let repo_id = crabllm_mlx::registry::resolve(model)
        .unwrap_or(model)
        .to_string();

    // Already cached?
    if std::path::Path::new(model).is_dir() || crabllm_mlx::cached_model_path(&repo_id).is_some() {
        return;
    }

    eprintln!("downloading {repo_id}...");
    let (tx, rx) = std::sync::mpsc::channel();
    let repo = repo_id.clone();
    let handle = std::thread::spawn(move || crabllm_mlx::download_model(&repo, &tx));

    // Print progress on the main thread.
    let mut current_file = String::new();
    let mut file_total: usize = 0;
    let mut file_downloaded: usize = 0;
    for event in rx {
        match event {
            DownloadEvent::FileStart {
                filename,
                total_bytes,
            } => {
                current_file = filename;
                file_total = total_bytes;
                file_downloaded = 0;
            }
            DownloadEvent::FileProgress { bytes } => {
                file_downloaded += bytes;
                if file_total > 0 {
                    let pct = file_downloaded * 100 / file_total;
                    let mb = file_downloaded / (1024 * 1024);
                    let total_mb = file_total / (1024 * 1024);
                    eprint!("\r  {current_file}: {mb}/{total_mb} MB ({pct}%)    ");
                }
            }
            DownloadEvent::FileDone => {
                eprintln!("\r  {current_file}: done                              ");
            }
            DownloadEvent::AllDone { .. } => {}
        }
    }

    match handle.join() {
        Ok(Ok(_)) => eprintln!("download complete."),
        Ok(Err(e)) => {
            eprintln!("download failed: {e}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("download thread panicked");
            std::process::exit(1);
        }
    }
}

#[tokio::main]
async fn main() {
    if std::env::var("RUST_LOG").is_err() {
        // SAFETY: called before any threads are spawned.
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    if cli.list {
        let models = crabllm_mlx::registry::list();
        eprintln!("{} predefined models:\n", models.len());
        for m in &models {
            let kind = match m.kind {
                crabllm_mlx::registry::ModelKind::Llm => "llm",
                crabllm_mlx::registry::ModelKind::Vlm => "vlm",
            };
            println!("{:<6} {:<45} {}", kind, m.alias, m.repo_id);
        }
        return;
    }

    let Some(model) = cli.model else {
        eprintln!("error: model name required. Use --list to see available models.");
        std::process::exit(1);
    };

    // Step 1: ensure downloaded (with progress).
    ensure_downloaded(&model);

    // Step 2: create pool + provider.
    let pool = match MlxPool::new(cli.idle_timeout) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("error: failed to create pool: {e}");
            std::process::exit(1);
        }
    };
    let provider = MlxProvider::new(pool);
    let mut history: Vec<Message> = Vec::new();

    eprintln!("loading model...");
    let warmup = ChatCompletionRequest {
        model: model.clone(),
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
            eprintln!("error: {e}");
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
            model: model.clone(),
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

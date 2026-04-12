//! End-to-end verification that VLM image input reaches the model.
//!
//! Downloads `mlx-community/Qwen3.5-0.8B-MLX-4bit` on first run
//! (~625 MB), then sends two requests that each include the repo's
//! `docs/book/favicon-8114d1fc.png` as an `image_url` content part:
//! one with a `file://` URL, one with a `data:image/png;base64,...`
//! URL. Prints both responses so a human can confirm the model saw
//! the image rather than falling back to a text-only non-sequitur.
//!
//! This is the minimal reproducer for MLX issue #54 — if this run
//! produces grounded responses, the `image_url` decode path in
//! `MessageParsing.swift` is working end-to-end for both inline and
//! filesystem inputs.
//!
//! Usage: cargo run -p crabllm-mlx --example vlm_smoke

use base64::{Engine, engine::general_purpose::STANDARD};
use crabllm_core::{ChatCompletionRequest, Message, Provider, Role};
use crabllm_mlx::{DownloadEvent, MlxPool, MlxProvider, cached_model_path, download_model};
use serde_json::{Value, json};
use std::{error::Error, sync::Arc};

const MODEL: &str = "mlx-community/Qwen3.5-0.8B-MLX-4bit";
const IMAGE_PATH: &str = "docs/book/favicon-8114d1fc.png";
const PROMPT: &str = "Describe this image in one short sentence.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var("RUST_LOG").is_err() {
        // SAFETY: called before any threads are spawned.
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    tracing_subscriber::fmt::init();

    ensure_downloaded()?;

    let image_abs = std::fs::canonicalize(IMAGE_PATH)?;
    let bytes = std::fs::read(&image_abs)?;
    let file_url = format!("file://{}", image_abs.display());
    let data_url = format!("data:image/png;base64,{}", STANDARD.encode(&bytes));

    let pool = Arc::new(MlxPool::new(1800)?);
    let provider = MlxProvider::new(pool);

    println!("=== model: {MODEL} ===");
    println!(
        "=== image: {} ({} bytes) ===",
        image_abs.display(),
        bytes.len()
    );
    println!();

    run_once(&provider, "file://", file_url).await?;
    println!();
    run_once(&provider, "data:image/png;base64", data_url).await?;

    Ok(())
}

fn ensure_downloaded() -> Result<(), Box<dyn Error>> {
    if cached_model_path(MODEL).is_some() {
        return Ok(());
    }
    eprintln!("downloading {MODEL}...");
    let (tx, rx) = std::sync::mpsc::channel();
    let repo = MODEL.to_string();
    let handle = std::thread::spawn(move || download_model(&repo, &tx));
    for event in rx {
        if let DownloadEvent::FileDone = event {
            // drain — progress noise not needed for a smoke test
        }
    }
    handle.join().expect("download thread panicked")?;
    eprintln!("download complete.");
    Ok(())
}

async fn run_once(
    provider: &MlxProvider,
    label: &str,
    url: String,
) -> Result<(), Box<dyn Error>> {
    let content: Value = json!([
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": url}},
    ]);

    let message = Message {
        role: Role::User,
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning_content: None,
        extra: Default::default(),
    };

    let request = ChatCompletionRequest {
        model: MODEL.to_string(),
        messages: vec![message],
        temperature: Some(0.0),
        top_p: None,
        max_tokens: Some(64),
        stream: None,
        stop: None,
        tools: None,
        tool_choice: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        user: None,
        reasoning_effort: None,
        extra: Default::default(),
    };

    println!("--- {label} ---");
    let response = provider.chat_completion(&request).await?;
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content_str())
        .unwrap_or("(empty)");
    println!("response: {text}");
    Ok(())
}

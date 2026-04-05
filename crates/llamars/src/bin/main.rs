use clap::{Parser, Subcommand};
use llamars::registry;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "llamars",
    about = "Managed llama.cpp server with Ollama registry"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an OpenAI-compatible server for one or more models
    Serve {
        /// Model names (e.g. llama3.2:3b qwen2.5:7b) or paths to GGUF files
        models: Vec<String>,

        /// Port to listen on (default: 8080)
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
    /// Pull a model from the Ollama registry
    Pull {
        /// Model name (e.g. llama3.2:3b)
        model: String,
    },
    /// List available tags for a model
    Tags {
        /// Model name (e.g. llama3.2)
        model: String,
    },
    /// Download the llama-server binary for this platform
    Download {
        /// Release tag (e.g. b4567). Defaults to latest.
        #[arg(short, long)]
        tag: Option<String>,
    },
    /// Check that llama-server is installed and reachable
    Check,
    /// Show the resolved path to the llama-server binary
    Which,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { models, port } => serve(models, port).await,
        Commands::Pull { model } => pull(&model),
        Commands::Tags { model } => tags(&model),
        Commands::Download { tag } => download(tag.as_deref()),
        Commands::Check => check(),
        Commands::Which => which(),
    }
}

async fn serve(models: Vec<String>, port: u16) {
    if models.is_empty() {
        eprintln!("error: at least one model is required");
        std::process::exit(1);
    }

    let bin = match llamars::find_server_binary() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let cache_dir = match registry::default_cache_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // Ensure all models are pulled.
    for model in &models {
        if std::path::Path::new(model).exists() {
            continue;
        }
        if registry::cached_model_path(model, &cache_dir).is_some() {
            continue;
        }
        let (name, tag) = registry::parse_model_name(model);
        eprintln!("pulling {name}:{tag}...");
        if let Err(e) = registry::pull_model(model, &cache_dir, &|_, _| {}) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }

    let pool = Arc::new(llamars::ServerPool::new(bin, cache_dir));
    pool.start_idle_monitor();

    let state = llamars::proxy::ProxyState {
        pool,
        client: reqwest::Client::new(),
        models,
    };

    let app = llamars::proxy::router(state);
    let addr = format!("0.0.0.0:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: failed to bind to {addr}: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("llamars listening on {addr}");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn pull(model: &str) {
    let cache_dir = match registry::default_cache_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let (name, tag) = registry::parse_model_name(model);
    eprintln!("pulling {name}:{tag}...");

    let last_pct = std::cell::Cell::new(0u8);
    match registry::pull_model(model, &cache_dir, &|downloaded, total| {
        if total == 0 {
            return;
        }
        let pct = (downloaded * 100 / total) as u8;
        if pct != last_pct.get() {
            last_pct.set(pct);
            let downloaded_mb = downloaded / (1024 * 1024);
            let total_mb = total / (1024 * 1024);
            eprint!("\r  {downloaded_mb} MB / {total_mb} MB ({pct}%)    ");
        }
    }) {
        Ok(path) => {
            eprintln!("\r  done: {}", path.display());
        }
        Err(e) => {
            eprintln!("\nerror: {e}");
            std::process::exit(1);
        }
    }
}

fn tags(model: &str) {
    let (name, _) = registry::parse_model_name(model);
    match registry::fetch_tags(name) {
        Ok(tags) => {
            for tag in &tags {
                println!("{name}:{tag}");
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn download(tag: Option<&str>) {
    match llamars::download(tag) {
        Ok(path) => {
            eprintln!("llama-server ready at {}", path.display());
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn check() {
    match llamars::find_server_binary() {
        Ok(path) => {
            eprintln!("llama-server found: {}", path.display());
            let output = std::process::Command::new(&path).arg("--version").output();
            match output {
                Ok(out) => {
                    let version = String::from_utf8_lossy(&out.stdout);
                    let version = version.trim();
                    if !version.is_empty() {
                        eprintln!("{version}");
                    } else {
                        let version = String::from_utf8_lossy(&out.stderr);
                        let version = version.trim();
                        if !version.is_empty() {
                            eprintln!("{version}");
                        }
                    }
                }
                Err(_) => eprintln!("(could not determine version)"),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn which() {
    match llamars::find_server_binary() {
        Ok(path) => println!("{}", path.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

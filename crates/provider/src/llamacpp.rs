use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use crabllm_core::Error;

/// Managed llama-server child process.
///
/// Owns the lifecycle of a `llama-server` binary: finding a free port,
/// spawning the process, waiting for it to become healthy, and tearing
/// it down on drop.
///
/// NOTE: port selection uses bind-to-0-then-release, which has an inherent
/// TOCTOU race. On a busy machine another process could grab the port between
/// release and llama-server's bind. This is the standard approach when the
/// child doesn't accept a pre-bound fd.
pub struct LlamaCppServer {
    child: Option<Child>,
    port: u16,
}

/// Configuration needed to spawn a llama-server process.
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub n_threads: Option<u32>,
}

impl LlamaCppServer {
    /// Spawn a llama-server process with the given config.
    ///
    /// Picks a free port, starts the process, and waits until the
    /// health endpoint responds (up to 120 seconds).
    pub fn spawn(bin: &Path, config: &LlamaCppConfig) -> Result<Self, Error> {
        let port = pick_free_port()?;

        let mut cmd = Command::new(bin);
        cmd.arg("--model")
            .arg(&config.model_path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(config.n_ctx.to_string())
            .arg("--n-gpu-layers")
            .arg(config.n_gpu_layers.to_string());

        if let Some(threads) = config.n_threads {
            cmd.arg("--threads").arg(threads.to_string());
        }

        // Stdout goes to /dev/null — llama-server writes everything useful
        // to stderr. Piping stdout without reading it would block the process
        // once the pipe buffer fills.
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| Error::Internal(format!("failed to spawn llama-server: {e}")))?;

        // Drain stderr in a background thread to prevent pipe buffer blocking
        // and forward logs to the gateway's stderr.
        let stderr = child.stderr.take();
        std::thread::spawn(move || {
            if let Some(stderr) = stderr {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(line) => eprintln!("[llama-server] {line}"),
                        Err(_) => break,
                    }
                }
            }
        });

        // Wait for the health endpoint, checking for early process death.
        if let Err(e) = wait_for_health(&mut child, port, Duration::from_secs(120)) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }

        Ok(Self {
            child: Some(child),
            port,
        })
    }

    /// The base URL for OpenAI-compatible requests.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    /// The port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Kill the child process and wait for it to exit.
    /// Safe to call multiple times — subsequent calls are no-ops.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for LlamaCppServer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Bind to port 0 to let the OS assign a free port, then release it.
fn pick_free_port() -> Result<u16, Error> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| Error::Internal(format!("failed to pick free port: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| Error::Internal(format!("failed to get local addr: {e}")))?
        .port();
    drop(listener);
    Ok(port)
}

/// Poll the /health endpoint until it returns 200, the child dies, or timeout.
fn wait_for_health(child: &mut Child, port: u16, timeout: Duration) -> Result<(), Error> {
    let url = format!("http://127.0.0.1:{port}/health");
    let start = std::time::Instant::now();
    let interval = Duration::from_millis(500);

    while start.elapsed() < timeout {
        // Check if the process died before becoming healthy.
        match child.try_wait() {
            Ok(Some(status)) => {
                return Err(Error::Internal(format!(
                    "llama-server exited during startup with {status}"
                )));
            }
            Err(e) => {
                return Err(Error::Internal(format!(
                    "failed to check llama-server status: {e}"
                )));
            }
            Ok(None) => {} // still running
        }

        if let Ok(resp) = ureq::get(&url).call() {
            if resp.status() == 200 {
                return Ok(());
            }
        }
        std::thread::sleep(interval);
    }

    Err(Error::Internal(format!(
        "llama-server on port {port} did not become healthy within {}s",
        timeout.as_secs()
    )))
}

/// Find the `llama-server` binary.
///
/// Search order:
/// 1. `$LLAMA_SERVER` environment variable
/// 2. `llama-server` on `$PATH`
pub fn find_server_binary() -> Result<PathBuf, Error> {
    if let Ok(path) = std::env::var("LLAMA_SERVER") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
        return Err(Error::Internal(format!(
            "LLAMA_SERVER={path} does not exist"
        )));
    }

    which::which("llama-server").map_err(|_| {
        Error::Internal(
            "llama-server not found on PATH. Install it or set LLAMA_SERVER env var".to_string(),
        )
    })
}

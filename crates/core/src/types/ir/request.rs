use crate::ir::{Content, Message};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub system: Option<Vec<Content>>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<Thinking>,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    Disabled,
    Named(String),
}

#[derive(Debug, Clone)]
pub struct Thinking {
    pub effort: Effort,
    /// An explicit budget when the client gave one, as Anthropic and Gemini
    /// take. `None` means "whatever `effort` implies".
    pub budget_tokens: Option<u32>,
}

impl Thinking {
    /// The thinking budget to send, clamped below `max_tokens` — Anthropic
    /// rejects a request whose budget meets or exceeds it.
    pub fn budget(&self, max_tokens: u32) -> u32 {
        self.budget_tokens
            .unwrap_or_else(|| self.effort.budget_tokens())
            .min(max_tokens.saturating_sub(1))
    }
}

/// How hard to think. These are OpenAI's `reasoning_effort` levels, which
/// double as the canonical scale: Anthropic and Gemini take an integer budget
/// instead, and [`Effort::budget_tokens`] is the bridge between the two.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    /// Budget for this level, following the table LiteLLM uses to bridge the
    /// same two representations.
    pub fn budget_tokens(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Minimal => 128,
            Self::Low => 1024,
            Self::Medium => 2048,
            Self::High => 4096,
            Self::XHigh => 8192,
            Self::Max => 16384,
        }
    }

    /// The strongest level a budget affords, rounding down — a budget of 8000
    /// buys `High` (4096), not `XHigh` (8192).
    pub fn from_budget(budget: u32) -> Self {
        [
            Self::Max,
            Self::XHigh,
            Self::High,
            Self::Medium,
            Self::Low,
            Self::Minimal,
        ]
        .into_iter()
        .find(|level| budget >= level.budget_tokens())
        .unwrap_or(Self::None)
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        })
    }
}

impl std::str::FromStr for Effort {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "none" => Self::None,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            _ => return Err(()),
        })
    }
}

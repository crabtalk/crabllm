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
    /// `None` means the client said nothing; `Some(Effort::None)` means it
    /// explicitly asked for no thinking.
    pub thinking: Option<Effort>,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    /// Marks this tool as the end of the cacheable tool prefix. Only the last
    /// entry needs it — there are four breakpoints and the whole declaration
    /// is stable across a conversation.
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    Disabled,
    Named(String),
}

/// How hard to think. The named levels are OpenAI's `reasoning_effort` scale,
/// which Anthropic's `output_config.effort` also speaks; [`Effort::Budget`]
/// carries the integer Gemini and Anthropic's pre-4.6 `thinking` dialect take.
/// A request arrives in one form or the other, never both — [`Effort::level`]
/// and [`Effort::budget_tokens`] convert to whichever the target wants.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    XHigh,
    Max,
    Budget(u32),
}

/// A budget of zero is how Gemini spells "no thinking".
impl From<u32> for Effort {
    fn from(tokens: u32) -> Self {
        if tokens == 0 {
            Self::None
        } else {
            Self::Budget(tokens)
        }
    }
}

impl Effort {
    /// The named level, rounding a budget down to the strongest it affords —
    /// 8000 buys `High` (4096), not `XHigh` (8192). Never returns `Budget`.
    pub fn level(self) -> Self {
        let Self::Budget(budget) = self else {
            return self;
        };
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

    /// The budget this effort implies, following the table LiteLLM uses to
    /// bridge the same two representations. Callers clamp to what the target
    /// accepts — Anthropic rejects a budget that meets `max_tokens`, Gemini
    /// enforces a per-model range instead.
    pub fn budget_tokens(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Minimal => 128,
            Self::Low => 1024,
            Self::Medium => 2048,
            Self::High => 4096,
            Self::XHigh => 8192,
            Self::Max => 16384,
            Self::Budget(tokens) => tokens,
        }
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.level() {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            // `level` never yields `Budget`, but listing it keeps the match
            // exhaustive so a new variant breaks the build instead of the wire.
            Self::None | Self::Budget(_) => "none",
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

use crate::Usage;
use crate::ir::StopReason;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallDelta(String),
    Usage(Usage),
    Stop(StopReason),
}

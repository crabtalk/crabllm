use crate::{Usage, ir::StopReason};
use alloc::string::String;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallDelta(String),
    Usage(Usage),
    Stop(StopReason),
}

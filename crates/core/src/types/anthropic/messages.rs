use crate::types::anthropic::{Content, ContentBlock, Message, ToolResultContent};
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

/// Operations on a wire-level message list. Lives as a trait so call sites
/// read as `messages.coalesce_tool_results()` rather than as free functions
/// taking `&mut Vec<Message>`.
pub trait Messages {
    /// Merge contiguous user messages whose blocks are entirely `tool_result`s
    /// into a single message. Parallel tool calls produce one user message per
    /// dispatch on the wire; Anthropic requires every `tool_use` block's
    /// matching `tool_result` to live in the SAME user message immediately
    /// following the assistant — splitting them violates that contract.
    fn coalesce_tool_results(&mut self);

    /// Synthesize `tool_result` blocks for any `tool_use` whose id has no
    /// matching `tool_result` in the immediately following user message.
    /// Orphans typically come from a cancelled dispatch that never committed
    /// its result. The synthetic result carries the text `"[interrupted]"`.
    /// Run AFTER [`coalesce_tool_results`] so the lookup sees every sibling
    /// `tool_result` in one place.
    ///
    /// [`coalesce_tool_results`]: Messages::coalesce_tool_results
    fn ensure_tool_pairing(&mut self);
}

impl Messages for Vec<Message> {
    fn coalesce_tool_results(&mut self) {
        let mut out: Vec<Message> = Vec::with_capacity(self.len());
        for msg in self.drain(..) {
            let mergeable = out.last().is_some_and(|prev| {
                prev.is_tool_result_only_user() && msg.is_tool_result_only_user()
            });
            if !mergeable {
                out.push(msg);
                continue;
            }
            let Some(target) = out.last_mut().and_then(|m| m.blocks_mut()) else {
                out.push(msg);
                continue;
            };
            if let Content::Blocks(source) = msg.content {
                target.extend(source);
            }
        }
        *self = out;
    }

    fn ensure_tool_pairing(&mut self) {
        // Pass 1: identify (assistant_index, synthetic_blocks). Read-only.
        let fixes: Vec<(usize, Vec<ContentBlock>)> = self
            .iter()
            .enumerate()
            .filter_map(|(i, msg)| {
                let next_results: alloc::collections::BTreeSet<&str> = self
                    .get(i + 1)
                    .map(|m| m.tool_result_ids().collect())
                    .unwrap_or_default();
                let missing: Vec<(String, String)> = msg
                    .tool_uses()
                    .filter(|(id, _)| !next_results.contains(id))
                    .map(|(id, name)| (id.to_string(), name.to_string()))
                    .collect();
                (!missing.is_empty()).then(|| {
                    let synthetic = missing
                        .into_iter()
                        .map(|(id, name)| ContentBlock::ToolResult {
                            tool_use_id: id,
                            name: Some(name),
                            content: ToolResultContent::Text("[interrupted]".to_string()),
                            cache_control: None,
                        })
                        .collect();
                    (i, synthetic)
                })
            })
            .collect();

        // Pass 2: apply in reverse so earlier indices don't shift when a
        // later index inserts a new message.
        for (i, synthetic) in fixes.into_iter().rev() {
            let next_is_user = self.get(i + 1).is_some_and(|m| m.role == "user");
            if !next_is_user {
                self.insert(
                    i + 1,
                    Message {
                        role: "user".to_string(),
                        content: Content::Blocks(synthetic),
                    },
                );
                continue;
            }
            let Some(blocks) = self[i + 1].blocks_mut() else {
                continue;
            };
            // tool_results precede any non-tool_result content in the user msg.
            let at = blocks
                .iter()
                .position(|b| !matches!(b, ContentBlock::ToolResult { .. }))
                .unwrap_or(blocks.len());
            blocks.splice(at..at, synthetic);
        }
    }
}

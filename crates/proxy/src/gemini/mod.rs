//! Gemini-compatible inbound endpoint.
//!
//! Handles `POST /v1beta/models/{model}:generateContent` and
//! `POST /v1beta/models/{model}:streamGenerateContent` — Google's wire shape.
//! Clients pointing their Gemini SDK at the proxy hit these routes; the proxy
//! either forwards the raw bytes (for `is_gemini_compat` providers) or
//! translates through the canonical Anthropic IR.

pub use handler::generate_content;

mod handler;

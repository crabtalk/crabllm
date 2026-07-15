//! The `bridge`-mode routing decision — pure, so it's tested directly without
//! a gateway. The dispatch it drives (native forward vs. IR translation) is
//! exercised end-to-end against a live proxy.
use crabllm_sdk::{Route, core::Dialect, route};

#[test]
fn anthropic_native_forwards() {
    assert_eq!(route(&[Dialect::Anthropic]), Route::Native);
}

#[test]
fn openai_only_translates() {
    assert_eq!(route(&[Dialect::Openai]), Route::Translate);
}

#[test]
fn native_wins_when_a_model_supports_both() {
    // Compat providers advertise both; fidelity means we forward, not translate.
    assert_eq!(route(&[Dialect::Openai, Dialect::Anthropic]), Route::Native);
}

#[test]
fn unknown_model_tries_native_then_translates() {
    // Empty = not in the catalog (or the catalog couldn't be fetched).
    assert_eq!(route(&[]), Route::NativeElseTranslate);
    // A dialect we can't serve over the Anthropic interface is also "unknown".
    assert_eq!(route(&[Dialect::Gemini]), Route::NativeElseTranslate);
}

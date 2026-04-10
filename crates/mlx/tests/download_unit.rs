//! Offline unit tests for `crabllm_mlx::download`. No network.
//!
//! With hf-hub backing the downloader, most cache/download logic is
//! tested upstream. We only verify our file-filter allowlist and the
//! error surface that our code adds on top.

#[test]
fn cached_path_returns_none_for_unknown_repo() {
    // A repo that was never downloaded returns None.
    assert!(crabllm_mlx::cached_model_path("crabllm-test/does-not-exist-xyz").is_none());
}

//! `validate_provider` is the single provider validator — the registry runs it
//! at build time and the proxy admin API runs it before accepting a provider.
//! These lock the rules that used to diverge between the two.
use crabllm_core::{ProviderConfig, ProviderKind};
use crabllm_provider::validate_provider;

fn cfg(
    kind: Option<ProviderKind>,
    api_key: Option<&str>,
    base_url: Option<&str>,
) -> ProviderConfig {
    ProviderConfig {
        kind,
        api_key: api_key.map(String::from),
        base_url: base_url.map(String::from),
        models: vec!["m".into()],
        ..Default::default()
    }
}

#[test]
fn anthropic_needs_api_key_not_just_base_url() {
    // The exact divergence the unification closed: a base_url-only Anthropic
    // provider was accepted by the admin path but rejected at registry build.
    let base_url_only = cfg(Some(ProviderKind::Anthropic), None, Some("https://x"));
    assert!(validate_provider("a", &base_url_only).is_err());

    let with_key = cfg(Some(ProviderKind::Anthropic), Some("k"), None);
    assert!(validate_provider("a", &with_key).is_ok());
}

#[test]
fn empty_models_is_rejected() {
    let mut c = cfg(Some(ProviderKind::Openai), Some("k"), None);
    c.models.clear();
    assert!(validate_provider("o", &c).is_err());
}

#[test]
fn compat_kind_needs_api_key_bare_custom_needs_base_url() {
    // "zai" resolves through the compat table (map-key default) → needs api_key.
    assert!(validate_provider("zai", &cfg(None, Some("k"), None)).is_ok());
    assert!(validate_provider("zai", &cfg(None, None, None)).is_err());

    // A self-defined kind that isn't in the table needs base_url.
    assert!(validate_provider("my-thing", &cfg(None, None, Some("https://x"))).is_ok());
    assert!(validate_provider("my-thing", &cfg(None, None, None)).is_err());
}

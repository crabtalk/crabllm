//! Offline unit tests for `crabllm_mlx::download`. No network.
//!
//! The live-network smoke test lives in `download_smoke.rs` behind
//! the `MLX_DOWNLOAD_SMOKE=1` env var.

use crabllm_mlx::{cached_model_path, default_cache_dir, download_model};
use std::fs;

fn scoped_tmp(tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("crabllm-mlx-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn default_cache_dir_is_under_crabtalk() {
    let dir = default_cache_dir().expect("home dir");
    assert!(
        dir.ends_with(".crabtalk/cache/mlx"),
        "got {}",
        dir.display()
    );
}

#[test]
fn cached_path_returns_none_for_empty_cache() {
    let tmp = scoped_tmp("cache-empty");
    assert!(cached_model_path("mlx-community/fake", &tmp).is_none());
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cached_path_returns_none_for_dir_without_sentinel() {
    let tmp = scoped_tmp("cache-nomodel");
    let repo = "mlx-community/fake";
    fs::create_dir_all(tmp.join(repo)).unwrap();
    // config.json alone is not enough — must have the sentinel.
    fs::write(tmp.join(repo).join("config.json"), "{}").unwrap();
    assert!(cached_model_path(repo, &tmp).is_none());
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cached_path_returns_dir_once_sentinel_exists() {
    let tmp = scoped_tmp("cache-ready");
    let repo = "mlx-community/fake";
    fs::create_dir_all(tmp.join(repo)).unwrap();
    fs::write(tmp.join(repo).join(".crabllm-mlx-complete"), b"").unwrap();
    let resolved = cached_model_path(repo, &tmp).expect("cache should resolve");
    assert_eq!(resolved, tmp.join(repo));
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn download_rejects_empty_repo_id() {
    let tmp = scoped_tmp("reject-empty");
    let err = download_model("", &tmp, &|_, _| {}).expect_err("empty repo must fail");
    assert!(format!("{err}").contains("invalid model repo"));
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn download_rejects_path_traversal() {
    let tmp = scoped_tmp("reject-traversal");
    let err =
        download_model("../../etc/passwd", &tmp, &|_, _| {}).expect_err("path traversal must fail");
    assert!(format!("{err}").contains("invalid model repo"));
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn download_rejects_single_segment() {
    let tmp = scoped_tmp("reject-single");
    let err =
        download_model("mlx-community", &tmp, &|_, _| {}).expect_err("single segment must fail");
    assert!(format!("{err}").contains("namespace/name"));
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn download_rejects_extra_segments() {
    let tmp = scoped_tmp("reject-triple");
    let err = download_model("mlx-community/foo/bar", &tmp, &|_, _| {})
        .expect_err("three segments must fail");
    assert!(format!("{err}").contains("namespace/name"));
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn download_rejects_bad_chars() {
    let tmp = scoped_tmp("reject-badchar");
    let err = download_model("mlx-community/foo bar", &tmp, &|_, _| {})
        .expect_err("space in name must fail");
    assert!(format!("{err}").contains("invalid model repo"));
    let _ = fs::remove_dir_all(&tmp);
}

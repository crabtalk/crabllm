//! Raw `extern "C"` bindings to `mlx/include/crabllm_mlx.h`.
//!
//! Layout must match the header exactly. The Swift side pins these
//! offsets via `_Static_assert` in `mlx/tests/smoke.c`; this module is
//! the Rust-side mirror of the same contract. If you edit the header,
//! update both places and rerun `make -C mlx test`.

use std::os::raw::{c_char, c_int, c_void};

pub type CrabllmMlxStatus = i32;

pub const CRABLLM_MLX_OK: CrabllmMlxStatus = 0;
pub const CRABLLM_MLX_ERR_INVALID_ARG: CrabllmMlxStatus = 1;
pub const CRABLLM_MLX_ERR_MODEL_LOAD: CrabllmMlxStatus = 2;
pub const CRABLLM_MLX_ERR_UNSUPPORTED_ARCH: CrabllmMlxStatus = 3;
pub const CRABLLM_MLX_ERR_GENERATE: CrabllmMlxStatus = 4;
pub const CRABLLM_MLX_ERR_UNKNOWN: CrabllmMlxStatus = 99;

#[repr(C)]
pub struct CrabllmMlxSession {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CrabllmMlxPool {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CrabllmMlxGenerateOptions {
    pub seed: u64,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

#[repr(C)]
pub struct CrabllmMlxGenerateRequest {
    pub messages_json: *const c_char,
    pub tools_json: *const c_char,
    pub options: CrabllmMlxGenerateOptions,
    pub cancel_flag: *const u32,
}

#[repr(C)]
pub struct CrabllmMlxGenerateResult {
    pub text: *mut c_char,
    pub tool_calls_json: *mut c_char,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub error: *mut c_char,
}

/// Layout mirror of `CrabllmMlxLoadedModel` in `crabllm_mlx.h`.
/// `memory_bytes` is `size_t` on the C side — `usize` here keeps
/// the ABI stable on 64-bit platforms (the only target we build).
#[repr(C)]
pub struct CrabllmMlxLoadedModel {
    pub name: *const c_char,
    pub memory_bytes: usize,
    pub last_used_unix: i64,
}

pub type CrabllmMlxTokenFn = unsafe extern "C" fn(*const c_char, *mut c_void) -> c_int;

#[allow(dead_code)] // Session-level FFI is used by tests and as an escape hatch
unsafe extern "C" {
    pub fn crabllm_mlx_session_new(
        model_dir_path: *const c_char,
        out_session: *mut *mut CrabllmMlxSession,
        out_error: *mut *mut c_char,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_session_free(session: *mut CrabllmMlxSession);

    pub fn crabllm_mlx_generate(
        session: *mut CrabllmMlxSession,
        request: *const CrabllmMlxGenerateRequest,
        result: *mut CrabllmMlxGenerateResult,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_generate_stream(
        session: *mut CrabllmMlxSession,
        request: *const CrabllmMlxGenerateRequest,
        token_cb: CrabllmMlxTokenFn,
        user_data: *mut c_void,
        result: *mut CrabllmMlxGenerateResult,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_result_free(result: *mut CrabllmMlxGenerateResult);

    pub fn crabllm_mlx_string_free(s: *mut c_char);

    // Pool functions
    pub fn crabllm_mlx_pool_new(
        idle_timeout_secs: u64,
        out_pool: *mut *mut CrabllmMlxPool,
        out_error: *mut *mut c_char,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_pool_free(pool: *mut CrabllmMlxPool);

    pub fn crabllm_mlx_pool_generate(
        pool: *mut CrabllmMlxPool,
        model_dir_path: *const c_char,
        request: *const CrabllmMlxGenerateRequest,
        result: *mut CrabllmMlxGenerateResult,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_pool_generate_stream(
        pool: *mut CrabllmMlxPool,
        model_dir_path: *const c_char,
        request: *const CrabllmMlxGenerateRequest,
        token_cb: CrabllmMlxTokenFn,
        user_data: *mut c_void,
        result: *mut CrabllmMlxGenerateResult,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_pool_evict(pool: *mut CrabllmMlxPool, model_dir_path: *const c_char) -> i32;

    pub fn crabllm_mlx_pool_stop_all(pool: *mut CrabllmMlxPool);

    pub fn crabllm_mlx_pool_list_loaded(
        pool: *mut CrabllmMlxPool,
        out_array: *mut *mut CrabllmMlxLoadedModel,
        out_count: *mut usize,
        out_error: *mut *mut c_char,
    ) -> CrabllmMlxStatus;

    pub fn crabllm_mlx_pool_loaded_free(array: *mut CrabllmMlxLoadedModel, count: usize);
}

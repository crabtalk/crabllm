#![allow(dead_code)]
//! `MlxPool` — Rust-side safe wrapper around the Swift pool FFI.
//!
//! The actual multi-model cache, idle eviction, and model loading live
//! in Swift (see `mlx/Sources/CrabllmMlx/Pool.swift`). This module is
//! a thin `NonNull` handle with `Send + Sync` and `Drop`.

use crate::ffi;
use crate::session::{
    OwnedRequest, ResultGuard, TrampolineState, copy_c_string_opt, take_owned_c_string, trampoline,
    translate_status,
};
use crabllm_core::Error;
use std::{
    ffi::{CStr, CString, c_char},
    os::raw::c_void,
    ptr,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// One entry in [`MlxPool::loaded_models`]'s inventory. `name` is the
/// local directory path the pool stores the slot under (what was
/// passed to `generate`). `memory_bytes` is a best-effort weight-file
/// footprint on disk — the weights dominate MLX's unified-memory
/// residency, so it's a stable proxy for "how big is this slot".
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub name: String,
    pub memory_bytes: u64,
    pub last_used: SystemTime,
}

/// Handle to a Swift-side multi-model pool.
pub struct MlxPool {
    inner: ptr::NonNull<ffi::CrabllmMlxPool>,
}

unsafe impl Send for MlxPool {}
unsafe impl Sync for MlxPool {}

impl MlxPool {
    /// Create a pool with idle eviction. `idle_timeout_secs == 0` uses
    /// the Swift default (30 min).
    pub fn new(idle_timeout_secs: u64) -> Result<Self, Error> {
        crate::metallib::ensure_metallib();
        let mut pool_ptr: *mut ffi::CrabllmMlxPool = ptr::null_mut();
        let mut err_ptr: *mut c_char = ptr::null_mut();
        let status =
            unsafe { ffi::crabllm_mlx_pool_new(idle_timeout_secs, &mut pool_ptr, &mut err_ptr) };
        if status == ffi::CRABLLM_MLX_OK {
            let inner = ptr::NonNull::new(pool_ptr).ok_or_else(|| {
                Error::Internal("mlx: pool_new OK but pointer is NULL".to_string())
            })?;
            Ok(MlxPool { inner })
        } else {
            let msg = unsafe { take_owned_c_string(err_ptr) };
            Err(translate_status(status, msg))
        }
    }

    /// Non-streaming generation through the pool.
    pub(crate) fn generate(
        &self,
        model_dir: &str,
        req: &crate::session::GenerateRequest<'_>,
    ) -> Result<crate::session::GenerateOutput, Error> {
        let model_c = CString::new(model_dir)
            .map_err(|_| Error::Internal("mlx: model_dir contains NUL byte".to_string()))?;
        let owned = OwnedRequest::new(req)?;
        let mut guard = ResultGuard::zeroed();
        let status = unsafe {
            ffi::crabllm_mlx_pool_generate(
                self.inner.as_ptr(),
                model_c.as_ptr(),
                owned.as_raw(),
                guard.as_mut_ptr(),
            )
        };
        if status == ffi::CRABLLM_MLX_OK {
            let text = unsafe { copy_c_string_opt(guard.inner.text)? }.ok_or_else(|| {
                Error::Internal("mlx: pool generate OK but result.text is NULL".to_string())
            })?;
            let tool_calls_json = unsafe { copy_c_string_opt(guard.inner.tool_calls_json)? };
            Ok(crate::session::GenerateOutput {
                text,
                tool_calls_json,
                prompt_tokens: guard.inner.prompt_tokens,
                completion_tokens: guard.inner.completion_tokens,
            })
        } else {
            let msg = unsafe { copy_c_string_opt(guard.inner.error) }
                .ok()
                .flatten()
                .unwrap_or_else(|| "(no error message from Swift)".to_string());
            Err(translate_status(status, msg))
        }
    }

    /// Streaming generation through the pool.
    pub(crate) fn generate_stream<F>(
        &self,
        model_dir: &str,
        req: &crate::session::GenerateRequest<'_>,
        mut on_token: F,
    ) -> Result<crate::session::StreamOutput, Error>
    where
        F: FnMut(&str) -> bool,
    {
        let model_c = CString::new(model_dir)
            .map_err(|_| Error::Internal("mlx: model_dir contains NUL byte".to_string()))?;
        let owned = OwnedRequest::new(req)?;

        let mut state = TrampolineState {
            cb: &mut on_token,
            panicked: false,
        };
        let mut guard = ResultGuard::zeroed();
        let status = unsafe {
            ffi::crabllm_mlx_pool_generate_stream(
                self.inner.as_ptr(),
                model_c.as_ptr(),
                owned.as_raw(),
                trampoline::<F>,
                &mut state as *mut TrampolineState<'_, F> as *mut c_void,
                guard.as_mut_ptr(),
            )
        };

        if state.panicked {
            return Err(Error::Internal(
                "mlx: token callback panicked during streaming".to_string(),
            ));
        }

        if status == ffi::CRABLLM_MLX_OK {
            let tool_calls_json = unsafe { copy_c_string_opt(guard.inner.tool_calls_json)? };
            Ok(crate::session::StreamOutput {
                tool_calls_json,
                prompt_tokens: guard.inner.prompt_tokens,
                completion_tokens: guard.inner.completion_tokens,
            })
        } else {
            let msg = unsafe { copy_c_string_opt(guard.inner.error) }
                .ok()
                .flatten()
                .unwrap_or_else(|| "(no error message from Swift)".to_string());
            Err(translate_status(status, msg))
        }
    }

    /// Evict a single model.
    pub(crate) fn evict(&self, model_dir: &str) {
        if let Ok(c) = CString::new(model_dir) {
            unsafe { ffi::crabllm_mlx_pool_evict(self.inner.as_ptr(), c.as_ptr()) };
        }
    }

    /// Evict all models and stop the idle monitor.
    pub(crate) fn stop_all(&self) {
        unsafe { ffi::crabllm_mlx_pool_stop_all(self.inner.as_ptr()) };
    }

    /// Snapshot the pool's loaded-model inventory.
    ///
    /// Each returned [`LoadedModel`] is a copy of the Swift-side slot
    /// at snapshot time. Concurrent generate / evict calls race
    /// cleanly against this — the Swift actor serializes the
    /// snapshot with every other mutation.
    pub fn loaded_models(&self) -> Result<Vec<LoadedModel>, Error> {
        let mut arr: *mut ffi::CrabllmMlxLoadedModel = ptr::null_mut();
        let mut count: usize = 0;
        let mut err: *mut c_char = ptr::null_mut();
        let status = unsafe {
            ffi::crabllm_mlx_pool_list_loaded(
                self.inner.as_ptr(),
                &mut arr,
                &mut count,
                &mut err,
            )
        };
        if status != ffi::CRABLLM_MLX_OK {
            let msg = unsafe { take_owned_c_string(err) };
            return Err(translate_status(status, msg));
        }

        // Empty inventory: Swift may leave arr NULL; count == 0.
        // Nothing to free and nothing to copy.
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let item = unsafe { &*arr.add(i) };
            let name = if item.name.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(item.name) }
                    .to_string_lossy()
                    .into_owned()
            };
            let last_used = UNIX_EPOCH + Duration::from_secs(item.last_used_unix.max(0) as u64);
            out.push(LoadedModel {
                name,
                memory_bytes: item.memory_bytes as u64,
                last_used,
            });
        }
        unsafe { ffi::crabllm_mlx_pool_loaded_free(arr, count) };
        Ok(out)
    }
}

impl Drop for MlxPool {
    fn drop(&mut self) {
        unsafe { ffi::crabllm_mlx_pool_free(self.inner.as_ptr()) };
    }
}

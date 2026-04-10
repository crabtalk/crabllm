//! Safe Rust wrapper around the Swift `CrabllmMlxSession` FFI.
//!
//! One `Session` = one loaded model. The Session is declared reentrant
//! at the API boundary (Swift serializes internally if mlx-swift-lm
//! requires it), so `&Session` is enough for `generate*` — no interior
//! mutex is needed on the Rust side.
//!
//! **Blocking contract.** Every `generate*` method is synchronous and
//! may run for seconds to minutes. Tokio callers MUST wrap calls in
//! `tokio::task::spawn_blocking` or equivalent; running directly on a
//! worker thread will stall the runtime.
//!
//! **Cancellation.**
//!   * [`Session::generate`] reads a `&AtomicU32` from the
//!     `cancel_flag` field of the request; Swift polls it with
//!     acquire semantics between steps. Callers MUST use
//!     `Ordering::Release` or stronger when storing the flag or the
//!     Swift-side acquire is meaningless.
//!   * [`Session::generate_stream`] signals stop by returning `true`
//!     from the user callback. This is strictly simpler than an atomic
//!     poll because the callback is synchronous.

#![allow(dead_code)] // Session is pub(crate); used by pool + C smoke tests
use crate::{ffi, metallib};
use crabllm_core::Error;
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_void},
    panic,
    path::Path,
    ptr,
    sync::atomic::AtomicU32,
};

/// Sampling / generation knobs. Zero / non-positive values mean
/// "Swift-side default".
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GenerateOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

/// Per-call input to [`Session::generate`] / [`Session::generate_stream`].
pub(crate) struct GenerateRequest<'a> {
    /// UTF-8 JSON array of OpenAI-shape chat messages. Must be non-empty.
    pub messages_json: &'a str,
    /// Optional UTF-8 JSON array of OpenAI-shape tool definitions.
    pub tools_json: Option<&'a str>,
    pub options: GenerateOptions,
    /// Optional cancel flag. Ignored by `generate_stream` — that uses
    /// the callback return value instead.
    pub cancel_flag: Option<&'a AtomicU32>,
}

/// Result of a non-streaming generation.
#[derive(Debug, Clone)]
pub(crate) struct GenerateOutput {
    pub text: String,
    pub tool_calls_json: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Result of a streaming generation (text is delivered through the
/// token callback; only metadata comes back here).
#[derive(Debug, Clone)]
pub(crate) struct StreamOutput {
    pub tool_calls_json: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Handle to a Swift-side MLX session.
///
/// Dropping the `Session` calls `crabllm_mlx_session_free`. That call
/// must not race any in-flight `generate*` on the same session. The
/// Rust type system cannot enforce this by itself — typical safe usage
/// is to keep every `Session` behind an `Arc` and only drop it after
/// every consumer of the `Arc` has returned. Users who expose a bare
/// `Session` and then drop it mid-flight get UB; don't do that.
pub(crate) struct Session {
    inner: ptr::NonNull<ffi::CrabllmMlxSession>,
}

// Swift declares the session reentrant; the inner pointer is thread-safe
// as long as we don't free it while a generate call is in flight.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    /// Load a model from a local directory containing the weights,
    /// `config.json`, and tokenizer files. The Swift side never touches
    /// the network — use `crate::download` for fetch.
    ///
    /// Blocking. On a tokio runtime, wrap the call in `spawn_blocking`.
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        metallib::ensure_metallib();
        let path = model_dir.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            Error::Internal(format!(
                "mlx: model_dir path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let c_path = CString::new(path_str)
            .map_err(|_| Error::Internal("mlx: model_dir contains NUL byte".to_string()))?;

        let mut session_ptr: *mut ffi::CrabllmMlxSession = ptr::null_mut();
        let mut err_ptr: *mut c_char = ptr::null_mut();
        let status = unsafe {
            ffi::crabllm_mlx_session_new(c_path.as_ptr(), &mut session_ptr, &mut err_ptr)
        };

        if status == ffi::CRABLLM_MLX_OK {
            let inner = ptr::NonNull::new(session_ptr).ok_or_else(|| {
                Error::Internal("mlx: session_new OK but pointer is NULL".to_string())
            })?;
            Ok(Session { inner })
        } else {
            let msg = unsafe { take_owned_c_string(err_ptr) };
            Err(translate_status(status, msg))
        }
    }

    /// Non-streaming chat completion. Blocking; see the module-level
    /// doc for the tokio caveat.
    pub fn generate(&self, req: &GenerateRequest<'_>) -> Result<GenerateOutput, Error> {
        let owned = OwnedRequest::new(req)?;
        let mut guard = ResultGuard::zeroed();
        // SAFETY: `owned` lives until the end of this function and
        // holds the CStrings backing `owned.raw.{messages_json,
        // tools_json}`. The guard owns the result struct for the same
        // span. `self.inner` is non-null by construction.
        //
        // Do not null any field of `guard.inner` after the call — the
        // Drop must see the same pointers Swift wrote so `result_free`
        // releases them exactly once.
        let status = unsafe {
            ffi::crabllm_mlx_generate(self.inner.as_ptr(), owned.as_raw(), guard.as_mut_ptr())
        };

        if status == ffi::CRABLLM_MLX_OK {
            // copy_c_string_opt only copies; the guard's Drop calls
            // result_free which releases the Swift-owned originals.
            let text = unsafe { copy_c_string_opt(guard.inner.text)? }.ok_or_else(|| {
                Error::Internal("mlx: generate OK but result.text is NULL".to_string())
            })?;
            let tool_calls_json = unsafe { copy_c_string_opt(guard.inner.tool_calls_json)? };
            Ok(GenerateOutput {
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

    /// Streaming chat completion. `on_token` is invoked once per
    /// decoded fragment on the calling thread; return `true` to stop
    /// generation early (returns `Ok` with partial metadata).
    ///
    /// Panics in `on_token` are caught and treated as an immediate
    /// stop; they do not unwind across the FFI boundary. The panic
    /// surfaces as an `Error::Internal`. Do not reuse the same closure
    /// across calls after a panic — its captured state is poisoned.
    pub fn generate_stream<F>(
        &self,
        req: &GenerateRequest<'_>,
        mut on_token: F,
    ) -> Result<StreamOutput, Error>
    where
        F: FnMut(&str) -> bool,
    {
        let owned = OwnedRequest::new(req)?;
        let mut state = TrampolineState {
            cb: &mut on_token,
            panicked: false,
        };
        let mut guard = ResultGuard::zeroed();

        // SAFETY: the FFI call is synchronous. `state` lives on this
        // stack frame and does not escape — Swift must not stash the
        // user_data pointer for delivery after the call returns. See
        // the contract in `mlx/include/crabllm_mlx.h`. `owned` keeps
        // the CStrings alive; `guard` owns the result on drop.
        let status = unsafe {
            ffi::crabllm_mlx_generate_stream(
                self.inner.as_ptr(),
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
            Ok(StreamOutput {
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
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { ffi::crabllm_mlx_session_free(self.inner.as_ptr()) };
    }
}

// ---------- internals ----------

/// Holds CStrings for the lifetime of a single FFI generate call.
///
/// Field order matters: `raw` contains raw pointers into `_messages`
/// and `_tools`. Drop runs in declaration order, so `raw` drops first
/// (trivially) and the CStrings drop after. Reordering fields, adding
/// a derive that moves state, or replacing either CString with a
/// borrowed reference will dangle `raw.messages_json` / `raw.tools_json`.
pub(crate) struct OwnedRequest {
    _messages: CString,
    _tools: Option<CString>,
    raw: ffi::CrabllmMlxGenerateRequest,
}

impl OwnedRequest {
    pub(crate) fn new(req: &GenerateRequest<'_>) -> Result<Self, Error> {
        let messages = CString::new(req.messages_json)
            .map_err(|_| Error::Internal("mlx: messages_json contains NUL byte".to_string()))?;
        let tools =
            match req.tools_json {
                Some(t) => Some(CString::new(t).map_err(|_| {
                    Error::Internal("mlx: tools_json contains NUL byte".to_string())
                })?),
                None => None,
            };

        let tools_ptr = tools.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());
        // AtomicU32::as_ptr returns *mut u32. Cast to *const u32 for
        // the FFI — Swift reads but never writes.
        let cancel_ptr = req
            .cancel_flag
            .map(|f| f.as_ptr() as *const u32)
            .unwrap_or(ptr::null());

        let raw = ffi::CrabllmMlxGenerateRequest {
            messages_json: messages.as_ptr(),
            tools_json: tools_ptr,
            options: ffi::CrabllmMlxGenerateOptions {
                seed: 0, // mlx-swift-lm has no seed knob; reserved in ABI
                max_tokens: req.options.max_tokens,
                temperature: req.options.temperature,
                top_p: req.options.top_p,
            },
            cancel_flag: cancel_ptr,
        };

        Ok(OwnedRequest {
            _messages: messages,
            _tools: tools,
            raw,
        })
    }

    pub(crate) fn as_raw(&self) -> *const ffi::CrabllmMlxGenerateRequest {
        &self.raw
    }
}

/// RAII guard around a `CrabllmMlxGenerateResult`. Drop calls
/// `crabllm_mlx_result_free` so any Swift-owned strings we did not
/// steal (on error paths, or on early return) are released exactly once.
pub(crate) struct ResultGuard {
    pub(crate) inner: ffi::CrabllmMlxGenerateResult,
}

impl ResultGuard {
    pub(crate) fn zeroed() -> Self {
        ResultGuard {
            inner: ffi::CrabllmMlxGenerateResult {
                text: ptr::null_mut(),
                tool_calls_json: ptr::null_mut(),
                prompt_tokens: 0,
                completion_tokens: 0,
                error: ptr::null_mut(),
            },
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::CrabllmMlxGenerateResult {
        &mut self.inner
    }
}

impl Drop for ResultGuard {
    fn drop(&mut self) {
        unsafe { ffi::crabllm_mlx_result_free(&mut self.inner) };
    }
}

/// State threaded through the streaming callback. The caller holds
/// `cb: &mut F` on the stack; the pointer is valid for the duration of
/// the synchronous FFI call and invalid immediately after.
pub(crate) struct TrampolineState<'cb, F: FnMut(&str) -> bool> {
    pub cb: &'cb mut F,
    pub panicked: bool,
}

pub(crate) extern "C" fn trampoline<F: FnMut(&str) -> bool>(
    token: *const c_char,
    user_data: *mut c_void,
) -> c_int {
    if user_data.is_null() {
        return 1;
    }
    // SAFETY: user_data was constructed by generate_stream as a
    // `*mut TrampolineState<'_, F>`. Monomorphization on F makes the
    // cast exact. The pointer is valid for the duration of the
    // synchronous FFI call — Swift must not stash it past return.
    let state = unsafe { &mut *(user_data as *mut TrampolineState<'_, F>) };
    if state.panicked {
        return 1;
    }
    if token.is_null() {
        return 0;
    }
    // SAFETY: header promises the token pointer is a valid NUL-
    // terminated C string for the duration of the callback.
    let slice = unsafe { CStr::from_ptr(token) };
    // The header promises UTF-8. If Swift violates that, stop the
    // stream and let the caller see a poisoned closure (panicked=false,
    // but the stream ends early) — a future edit may surface this as
    // a distinct status.
    let s = match slice.to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };
    let cb = &mut state.cb;
    // AssertUnwindSafe: if F panics, its captured state may be torn.
    // The caller is told not to reuse the closure after this returns
    // via the module-level docs.
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| (*cb)(s)));
    match result {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => {
            state.panicked = true;
            1
        }
    }
}

/// Take ownership of a C string allocated standalone by Swift (via
/// `out_error` on `session_new`). Frees the original with
/// `crabllm_mlx_string_free` after copying.
///
/// Returns a placeholder on NULL input so callers can unconditionally
/// use the returned string in an error message.
pub(crate) unsafe fn take_owned_c_string(ptr: *mut c_char) -> String {
    if ptr.is_null() {
        return "(no error message from Swift)".to_string();
    }
    // to_string_lossy is deliberate here: the error message path must
    // not fail even if Swift handed us a non-UTF-8 buffer. The
    // success paths use strict UTF-8 via `copy_c_string_opt`.
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { ffi::crabllm_mlx_string_free(ptr) };
    s
}

/// Copy a C string field from a `ResultGuard` without freeing it.
/// Strict UTF-8: invalid bytes are an error, not a silent U+FFFD
/// substitution. Returns `Ok(None)` on NULL.
pub(crate) unsafe fn copy_c_string_opt(ptr: *mut c_char) -> Result<Option<String>, Error> {
    if ptr.is_null() {
        return Ok(None);
    }
    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes();
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(Some(s.to_string())),
        Err(_) => Err(Error::Internal(
            "mlx: swift returned non-UTF-8 bytes in a result field".to_string(),
        )),
    }
}

pub(crate) fn translate_status(status: ffi::CrabllmMlxStatus, msg: String) -> Error {
    let label = match status {
        ffi::CRABLLM_MLX_ERR_INVALID_ARG => "mlx invalid arg",
        ffi::CRABLLM_MLX_ERR_MODEL_LOAD => "mlx model load",
        ffi::CRABLLM_MLX_ERR_UNSUPPORTED_ARCH => "mlx unsupported architecture",
        ffi::CRABLLM_MLX_ERR_GENERATE => "mlx generate",
        ffi::CRABLLM_MLX_ERR_UNKNOWN => "mlx unknown error",
        _ => "mlx unexpected status",
    };
    Error::Internal(format!("{label}: {msg}"))
}

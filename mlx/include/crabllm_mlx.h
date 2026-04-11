/*
 * crabllm_mlx.h — C ABI between Rust (crates/mlx) and Swift (mlx/).
 *
 * All strings are UTF-8. The Rust caller owns `CrabllmMlxGenerateRequest`
 * and all strings it points to. The Swift callee owns every string and
 * pointer it writes into `CrabllmMlxGenerateResult`; the caller must
 * release the result with `crabllm_mlx_result_free` whether the call
 * succeeded or failed.
 *
 * Ownership rules for `CrabllmMlxGenerateResult`:
 *
 *   - On success (CRABLLM_MLX_OK):
 *       `text`              non-NULL (may be empty) for non-streaming;
 *                           always NULL for streaming (delivered via
 *                           the token callback instead).
 *       `tool_calls_json`   non-NULL only if the model emitted tool calls.
 *       `prompt_tokens`     valid.
 *       `completion_tokens` valid.
 *       `error`             NULL.
 *   - On failure (anything else):
 *       `text`              NULL.
 *       `tool_calls_json`   NULL.
 *       `prompt_tokens`     0.
 *       `completion_tokens` 0.
 *       `error`             non-NULL, human-readable UTF-8.
 *
 * Thread-safety contract:
 *
 *   - A CrabllmMlxSession is reentrant: multiple threads may call
 *     crabllm_mlx_generate / crabllm_mlx_generate_stream concurrently
 *     against the same session. Swift serializes internally if required
 *     by mlx-swift-lm. The GPU is the bottleneck regardless.
 *   - crabllm_mlx_session_free must not race any in-flight generate call
 *     on the same session. The Rust wrapper keeps sessions alive behind
 *     an Arc so this is structurally impossible to violate.
 *   - Cancellation:
 *       * Non-streaming: `cancel_flag` points to a uint32 the Rust caller
 *         updates; Swift reads it with acquire semantics between steps.
 *         Non-zero stops generation and still returns the text produced
 *         so far as a successful result.
 *       * Streaming: the per-token callback returns non-zero to stop.
 *         There is no separate cancel flag — the callback is synchronous
 *         and knows whether the Rust stream consumer has dropped the
 *         receiver, so an extra atomic is noise.
 */

#ifndef CRABLLM_MLX_H
#define CRABLLM_MLX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque session handle. One instance = one loaded model. */
typedef struct CrabllmMlxSession CrabllmMlxSession;

/* Status is returned as a fixed-width int32 so the ABI is stable across
 * compilers that disagree on enum storage. The constants below are the
 * only legal values; anything else is a Swift-side bug. */
typedef int32_t CrabllmMlxStatus;

enum {
    CRABLLM_MLX_OK = 0,
    CRABLLM_MLX_ERR_INVALID_ARG = 1,
    CRABLLM_MLX_ERR_MODEL_LOAD = 2,
    CRABLLM_MLX_ERR_UNSUPPORTED_ARCH = 3,
    CRABLLM_MLX_ERR_GENERATE = 4,
    CRABLLM_MLX_ERR_UNKNOWN = 99
};

/* Sampling / generation knobs. Zero/negative values fall back to Swift-side
 * defaults so the Rust caller can leave request fields unset. Field order
 * is chosen to eliminate padding: 8, 4, 4, 4.
 *
 * `seed` is currently IGNORED by the mlx-swift-lm 2.31.3 backend —
 * GenerateParameters has no seed knob. The field is kept in the ABI so
 * callers can set it and a future MLX version can plumb it through
 * without breaking layout. Rust callers that rely on deterministic
 * output should not assume the seed takes effect yet. */
typedef struct {
    uint64_t seed;         /* currently ignored by mlx-swift-lm; see note above */
    uint32_t max_tokens;   /* 0 = Swift default (model max) */
    float temperature;     /* <= 0 = Swift default */
    float top_p;           /* <= 0 = Swift default */
} CrabllmMlxGenerateOptions;

/* Per-call input. All pointers are borrowed from the Rust caller for the
 * duration of the generate*() call only. */
typedef struct {
    /* UTF-8 JSON array of OpenAI-shape chat messages. Never NULL. */
    const char *messages_json;
    /* UTF-8 JSON array of OpenAI-shape tool definitions, or NULL. */
    const char *tools_json;
    CrabllmMlxGenerateOptions options;
    /* Pointer to a uint32 cancel flag owned by the caller. NULL disables
     * cancellation. Swift reads it with acquire semantics between steps.
     * Not used by crabllm_mlx_generate_stream — that uses the callback
     * return value instead. */
    const uint32_t *cancel_flag;
} CrabllmMlxGenerateRequest;

/* Per-call output. All pointer fields are owned by Swift and must be
 * released with crabllm_mlx_result_free. The caller zero-initializes the
 * struct before passing it in. */
typedef struct {
    char *text;
    char *tool_calls_json;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    char *error;
} CrabllmMlxGenerateResult;

/* Per-token callback for streaming generation.
 *
 * `token` is a freshly decoded UTF-8 fragment owned by Swift; valid only
 * for the duration of the callback. Copy it if you need to keep it.
 * `user_data` is the opaque pointer passed into generate_stream().
 *
 * Return value:
 *   0      — continue generating.
 *   non-0  — stop generating. The session finalizes the stream, fills
 *            out the result (tool_calls_json, token counts), and returns
 *            CRABLLM_MLX_OK.
 *
 * The callback runs synchronously on the generate_stream thread. It must
 * not call back into the FFI.
 */
typedef int (*CrabllmMlxTokenFn)(const char *token, void *user_data);

/*
 * Load a model from a local directory that already contains the weights,
 * config.json, and tokenizer files. Downloading is the Rust caller's job
 * (see crates/mlx/src/download.rs); this function never touches the
 * network.
 *
 * On success, *out_session is set to a new handle the caller owns.
 * On failure, returns a non-OK status and writes a human-readable error
 * string to *out_error (which the caller must release with
 * crabllm_mlx_string_free). CRABLLM_MLX_ERR_UNSUPPORTED_ARCH is returned
 * specifically when the config.json model_type is not registered in
 * mlx-swift-lm's LLMTypeRegistry / VLMTypeRegistry; the error string
 * contains the offending architecture name.
 *
 * Blocking. Call from a background thread.
 */
CrabllmMlxStatus crabllm_mlx_session_new(
    const char *model_dir_path,
    CrabllmMlxSession **out_session,
    char **out_error);

/*
 * Release a session. Safe to call with NULL. Must not race any in-flight
 * generate call against the same session.
 */
void crabllm_mlx_session_free(CrabllmMlxSession *session);

/*
 * Non-streaming generation. Writes `*result` in the shape documented at
 * the top of this file. Blocking; call from a background thread.
 *
 * Cancellation via request->cancel_flag stops generation early but the
 * call still returns CRABLLM_MLX_OK and result->text contains whatever
 * was produced up to the interrupt point.
 */
CrabllmMlxStatus crabllm_mlx_generate(
    CrabllmMlxSession *session,
    const CrabllmMlxGenerateRequest *request,
    CrabllmMlxGenerateResult *result);

/*
 * Streaming generation. Tokens are delivered through `token_cb` as they
 * are produced; `result->text` is always NULL on return. Tool calls (if
 * any) and token counts are written to `result` once generation ends.
 *
 * Cancellation is signalled by the callback returning non-zero. See the
 * CrabllmMlxTokenFn doc comment. A callback-terminated stream still
 * returns CRABLLM_MLX_OK.
 *
 * Blocking. Call from a background thread.
 */
CrabllmMlxStatus crabllm_mlx_generate_stream(
    CrabllmMlxSession *session,
    const CrabllmMlxGenerateRequest *request,
    CrabllmMlxTokenFn token_cb,
    void *user_data,
    CrabllmMlxGenerateResult *result);

/*
 * Release a result. Frees `text`, `tool_calls_json`, and `error` if
 * non-NULL, then zeroes the struct. Safe to call with NULL.
 */
void crabllm_mlx_result_free(CrabllmMlxGenerateResult *result);

/*
 * Free a standalone string returned by crabllm_mlx_session_new via
 * *out_error. Safe to call with NULL. Prefer crabllm_mlx_result_free for
 * generate*() outputs.
 */
void crabllm_mlx_string_free(char *s);

/* ── Multi-model pool ──
 *
 * The pool manages multiple loaded models keyed by local directory
 * path. Models are loaded on first request and evicted after the idle
 * timeout. Swift owns the lifecycle (native async, actor-isolated);
 * the Rust side only holds the opaque handle and calls through.
 *
 * Thread-safety: the pool is actor-isolated in Swift. All FFI entry
 * points are safe to call from any thread. Internally they bridge to
 * the actor via `blockingAwait` (same contract as session functions:
 * caller thread must not be a cooperative executor worker).
 */

typedef struct CrabllmMlxPool CrabllmMlxPool;

/*
 * Create a pool. `idle_timeout_secs == 0` uses the default (30 min).
 * Blocking. Call from a background thread.
 */
CrabllmMlxStatus crabllm_mlx_pool_new(
    uint64_t idle_timeout_secs,
    CrabllmMlxPool **out_pool,
    char **out_error);

/* Release a pool and all loaded models. Safe to call with NULL. */
void crabllm_mlx_pool_free(CrabllmMlxPool *pool);

/*
 * Non-streaming generation through the pool. The pool loads the model
 * at `model_dir_path` on first call and caches it for subsequent
 * requests. Semantics otherwise identical to crabllm_mlx_generate.
 * Blocking. Call from a background thread.
 */
CrabllmMlxStatus crabllm_mlx_pool_generate(
    CrabllmMlxPool *pool,
    const char *model_dir_path,
    const CrabllmMlxGenerateRequest *request,
    CrabllmMlxGenerateResult *result);

/*
 * Streaming generation through the pool. Semantics identical to
 * crabllm_mlx_generate_stream but model loading is pool-managed.
 * Blocking. Call from a background thread.
 */
CrabllmMlxStatus crabllm_mlx_pool_generate_stream(
    CrabllmMlxPool *pool,
    const char *model_dir_path,
    const CrabllmMlxGenerateRequest *request,
    CrabllmMlxTokenFn token_cb,
    void *user_data,
    CrabllmMlxGenerateResult *result);

/* Evict a single model from the pool. No-op if not loaded. */
void crabllm_mlx_pool_evict(CrabllmMlxPool *pool, const char *model_dir_path);

/* Evict all models and stop the idle monitor. */
void crabllm_mlx_pool_stop_all(CrabllmMlxPool *pool);

/*
 * One entry in the pool's loaded-model inventory. Ownership: returned
 * by crabllm_mlx_pool_list_loaded in a caller-owned array that must
 * be released with crabllm_mlx_pool_loaded_free. `name` is a null-
 * terminated UTF-8 string (the local directory path the pool stores
 * the slot under). `memory_bytes` is a best-effort resident footprint
 * — currently the sum of weight-file sizes on disk (.safetensors /
 * .bin / .gguf), which dominates MLX's unified-memory footprint.
 * `last_used_unix` is seconds since the epoch of the last generate
 * call that touched the slot.
 *
 * Field order is chosen to eliminate padding on 64-bit: 8, 8, 8.
 */
typedef struct {
    const char *name;
    size_t memory_bytes;
    int64_t last_used_unix;
} CrabllmMlxLoadedModel;

/*
 * Snapshot the pool's loaded-model inventory.
 *
 * On success, `*out_array` points to a newly allocated array of
 * `*out_count` `CrabllmMlxLoadedModel` entries; the caller owns the
 * array and every `name` pointer inside it, and must release the
 * whole thing via `crabllm_mlx_pool_loaded_free`. On empty pool,
 * `*out_count == 0` and `*out_array` may be NULL.
 *
 * Blocking. Call from a background thread. Actor-isolated: races
 * cleanly against concurrent generate / evict calls.
 */
CrabllmMlxStatus crabllm_mlx_pool_list_loaded(
    CrabllmMlxPool *pool,
    CrabllmMlxLoadedModel **out_array,
    size_t *out_count,
    char **out_error);

/*
 * Release the array returned by crabllm_mlx_pool_list_loaded. Frees
 * every `name` pointer and the array itself. Safe to call with
 * array == NULL and count == 0.
 */
void crabllm_mlx_pool_loaded_free(CrabllmMlxLoadedModel *array, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* CRABLLM_MLX_H */

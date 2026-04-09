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
 * is chosen to eliminate padding: 8, 4, 4, 4. */
typedef struct {
    uint64_t seed;         /* 0 = non-deterministic */
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

#ifdef __cplusplus
}
#endif

#endif /* CRABLLM_MLX_H */

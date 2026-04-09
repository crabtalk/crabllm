/*
 * smoke.c — Phase 2 ABI round-trip test.
 *
 * Run via `make -C mlx test` after `swift build -c release`.
 * Phase 3's build.rs will also exercise the same symbols through
 * Rust once the Rust crate lands.
 *
 * Verifies:
 *   1. The struct layout and status codes the Swift side assumes
 *      (see mlx/Sources/CrabllmMlx/Session.swift) match the C header.
 *      Any drift on either side fails _Static_assert below and breaks
 *      the build.
 *   2. session_new accepts a valid path and rejects NULL/empty.
 *   3. generate fills text + token counts, echoes the messages prefix.
 *   4. generate_stream fires four canned tokens and honors callback
 *      return-to-stop.
 *   5. result_free / string_free / session_free run cleanly with NULL.
 *
 * TODO(phase5): "/tmp/dummy-model" stops working the moment Phase 5
 * lands because the real session_new will stat the directory and load
 * weights. Update this test to point at a cached mlx-community model
 * when Phase 5 merges.
 */

#include "../include/crabllm_mlx.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------- Layout pin: anything here drifting breaks the build. --------
 *
 * The Swift side in Session.swift walks these structs via raw-pointer
 * offsets matching the numbers below. Keep both in sync or the FFI
 * silently reads garbage.
 */

_Static_assert(sizeof(CrabllmMlxGenerateOptions) == 24,
               "CrabllmMlxGenerateOptions must be 24 bytes (8+4+4+4+pad4)");
_Static_assert(offsetof(CrabllmMlxGenerateOptions, seed) == 0,
               "options.seed must be at offset 0");
_Static_assert(offsetof(CrabllmMlxGenerateOptions, max_tokens) == 8,
               "options.max_tokens must be at offset 8");
_Static_assert(offsetof(CrabllmMlxGenerateOptions, temperature) == 12,
               "options.temperature must be at offset 12");
_Static_assert(offsetof(CrabllmMlxGenerateOptions, top_p) == 16,
               "options.top_p must be at offset 16");

_Static_assert(sizeof(CrabllmMlxGenerateRequest) == 48,
               "CrabllmMlxGenerateRequest must be 48 bytes");
_Static_assert(offsetof(CrabllmMlxGenerateRequest, messages_json) == 0,
               "request.messages_json must be at offset 0");
_Static_assert(offsetof(CrabllmMlxGenerateRequest, tools_json) == 8,
               "request.tools_json must be at offset 8");
_Static_assert(offsetof(CrabllmMlxGenerateRequest, options) == 16,
               "request.options must be at offset 16");
_Static_assert(offsetof(CrabllmMlxGenerateRequest, cancel_flag) == 40,
               "request.cancel_flag must be at offset 40");

_Static_assert(sizeof(CrabllmMlxGenerateResult) == 32,
               "CrabllmMlxGenerateResult must be 32 bytes (8+8+4+4+8)");
_Static_assert(offsetof(CrabllmMlxGenerateResult, text) == 0,
               "result.text must be at offset 0");
_Static_assert(offsetof(CrabllmMlxGenerateResult, tool_calls_json) == 8,
               "result.tool_calls_json must be at offset 8");
_Static_assert(offsetof(CrabllmMlxGenerateResult, prompt_tokens) == 16,
               "result.prompt_tokens must be at offset 16");
_Static_assert(offsetof(CrabllmMlxGenerateResult, completion_tokens) == 20,
               "result.completion_tokens must be at offset 20");
_Static_assert(offsetof(CrabllmMlxGenerateResult, error) == 24,
               "result.error must be at offset 24");

/* Status constant pins — Swift mirrors these as `private let` in
 * Session.swift. If the header renumbers them, this file stops
 * compiling and Swift needs the matching update. */
_Static_assert(CRABLLM_MLX_OK == 0, "OK must be 0");
_Static_assert(CRABLLM_MLX_ERR_INVALID_ARG == 1, "INVALID_ARG must be 1");
_Static_assert(CRABLLM_MLX_ERR_MODEL_LOAD == 2, "MODEL_LOAD must be 2");
_Static_assert(CRABLLM_MLX_ERR_UNSUPPORTED_ARCH == 3, "UNSUPPORTED_ARCH must be 3");
_Static_assert(CRABLLM_MLX_ERR_GENERATE == 4, "GENERATE must be 4");
_Static_assert(CRABLLM_MLX_ERR_UNKNOWN == 99, "UNKNOWN must be 99");

/* ---------------------------- Runtime tests --------------------------- */

#define FAIL(fmt, ...) do { \
    fprintf(stderr, "smoke: " fmt "\n", ##__VA_ARGS__); \
    return 1; \
} while (0)

static int g_token_count = 0;
static int g_stop_after = -1;

static int token_cb(const char *token, void *user_data) {
    (void)user_data;
    if (token == NULL) {
        FAIL("token_cb received NULL token — stub should never emit NULL");
    }
    g_token_count++;
    if (g_stop_after > 0 && g_token_count >= g_stop_after) {
        return 1;
    }
    return 0;
}

static int test_session_new_rejects_null(void) {
    CrabllmMlxSession *session = NULL;
    char *err = NULL;
    CrabllmMlxStatus status = crabllm_mlx_session_new(NULL, &session, &err);
    if (status == CRABLLM_MLX_OK) {
        FAIL("session_new(NULL) should have failed");
    }
    if (err == NULL) {
        FAIL("session_new(NULL) did not populate out_error");
    }
    crabllm_mlx_string_free(err);
    return 0;
}

static int test_happy_path(void) {
    CrabllmMlxSession *session = NULL;
    char *err = NULL;
    if (crabllm_mlx_session_new("/tmp/dummy-model", &session, &err) != CRABLLM_MLX_OK) {
        FAIL("session_new failed: %s", err ? err : "(no error)");
    }
    if (session == NULL) {
        FAIL("session_new returned OK but out_session is NULL");
    }

    CrabllmMlxGenerateRequest request = {0};
    request.messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]";

    CrabllmMlxGenerateResult result = {0};
    if (crabllm_mlx_generate(session, &request, &result) != CRABLLM_MLX_OK) {
        FAIL("generate failed: %s", result.error ? result.error : "(no error)");
    }
    if (result.text == NULL) {
        FAIL("generate result.text is NULL on OK");
    }
    if (strstr(result.text, "dummy:") == NULL) {
        FAIL("generate result.text missing dummy prefix: %s", result.text);
    }
    if (result.prompt_tokens == 0 || result.completion_tokens == 0) {
        FAIL("generate result token counts are zero");
    }
    crabllm_mlx_result_free(&result);

    g_token_count = 0;
    g_stop_after = -1;
    CrabllmMlxGenerateResult stream_result = {0};
    if (crabllm_mlx_generate_stream(session, &request, token_cb, NULL, &stream_result) != CRABLLM_MLX_OK) {
        FAIL("generate_stream failed: %s",
             stream_result.error ? stream_result.error : "(no error)");
    }
    if (stream_result.text != NULL) {
        FAIL("generate_stream wrote text (should only stream via callback)");
    }
    if (g_token_count != 4) {
        FAIL("generate_stream fired %d tokens, expected 4", g_token_count);
    }
    crabllm_mlx_result_free(&stream_result);

    g_token_count = 0;
    g_stop_after = 2;
    CrabllmMlxGenerateResult cancel_result = {0};
    if (crabllm_mlx_generate_stream(session, &request, token_cb, NULL, &cancel_result) != CRABLLM_MLX_OK) {
        FAIL("generate_stream(stop=2) did not return OK on callback abort");
    }
    if (g_token_count != 2) {
        FAIL("generate_stream fired %d tokens after stop=2, expected 2", g_token_count);
    }
    crabllm_mlx_result_free(&cancel_result);

    crabllm_mlx_session_free(session);
    crabllm_mlx_session_free(NULL);  /* no-op safety check */
    crabllm_mlx_string_free(NULL);   /* no-op safety check */
    crabllm_mlx_result_free(NULL);   /* no-op safety check */
    return 0;
}

int main(void) {
    if (test_session_new_rejects_null() != 0) return 1;
    if (test_happy_path() != 0) return 1;
    printf("smoke: ok\n");
    return 0;
}

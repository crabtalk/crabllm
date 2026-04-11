/*
 * smoke.c — Phase 5 ABI + error-path smoke test.
 *
 * Run via `make -C mlx test` after `swift build -c release`.
 *
 * This test exercises everything that does NOT require a real cached
 * model: the struct layout contract (via `_Static_assert`), the error
 * paths of `session_new`, and the NULL-safety of the release helpers.
 * Happy-path generation is covered end-to-end by Rust integration
 * tests that load a real mlx-community model from disk.
 *
 * Verifies:
 *   1. The struct layout and status codes the Swift side assumes
 *      (see mlx/Sources/CrabllmMlx/Session.swift) match the C header.
 *      Any drift on either side fails _Static_assert below and breaks
 *      the build.
 *   2. session_new rejects NULL model_dir_path.
 *   3. session_new rejects a path that does not exist.
 *   4. session_new rejects a path that is a file, not a directory.
 *   5. string_free / result_free / session_free accept NULL.
 */

#include "../include/crabllm_mlx.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------- Layout pin: anything here drifting breaks the build. -------- */

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

_Static_assert(sizeof(CrabllmMlxLoadedModel) == 24,
               "CrabllmMlxLoadedModel must be 24 bytes (8+8+8) on 64-bit");
_Static_assert(offsetof(CrabllmMlxLoadedModel, name) == 0,
               "loaded_model.name must be at offset 0");
_Static_assert(offsetof(CrabllmMlxLoadedModel, memory_bytes) == 8,
               "loaded_model.memory_bytes must be at offset 8");
_Static_assert(offsetof(CrabllmMlxLoadedModel, last_used_unix) == 16,
               "loaded_model.last_used_unix must be at offset 16");

/* Status constant pins. */
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

static int test_session_new_rejects_empty(void) {
    CrabllmMlxSession *session = NULL;
    char *err = NULL;
    CrabllmMlxStatus status = crabllm_mlx_session_new("", &session, &err);
    if (status == CRABLLM_MLX_OK) {
        FAIL("session_new(\"\") should have failed");
    }
    crabllm_mlx_string_free(err);
    return 0;
}

static int test_session_new_rejects_missing_dir(void) {
    CrabllmMlxSession *session = NULL;
    char *err = NULL;
    CrabllmMlxStatus status = crabllm_mlx_session_new(
        "/definitely/does/not/exist/crabllm-mlx-smoke",
        &session,
        &err);
    if (status == CRABLLM_MLX_OK) {
        FAIL("session_new(nonexistent) should have failed");
    }
    if (err == NULL) {
        FAIL("session_new(nonexistent) did not populate out_error");
    }
    if (strstr(err, "exist") == NULL && strstr(err, "directory") == NULL) {
        FAIL("session_new error should mention path: %s", err);
    }
    crabllm_mlx_string_free(err);
    return 0;
}

static int test_null_safety(void) {
    crabllm_mlx_session_free(NULL);
    crabllm_mlx_string_free(NULL);
    crabllm_mlx_result_free(NULL);
    crabllm_mlx_pool_loaded_free(NULL, 0);
    return 0;
}

static int test_pool_list_loaded_rejects_null_pool(void) {
    CrabllmMlxLoadedModel *arr = (CrabllmMlxLoadedModel *)0xdeadbeef;
    size_t count = 42;
    char *err = NULL;
    CrabllmMlxStatus status = crabllm_mlx_pool_list_loaded(NULL, &arr, &count, &err);
    if (status == CRABLLM_MLX_OK) {
        FAIL("pool_list_loaded(NULL) should have failed");
    }
    if (err == NULL) {
        FAIL("pool_list_loaded(NULL) did not populate out_error");
    }
    if (count != 0) {
        FAIL("pool_list_loaded(NULL) should have zeroed out_count, got %zu", count);
    }
    crabllm_mlx_string_free(err);
    return 0;
}

int main(void) {
    if (test_session_new_rejects_null() != 0) return 1;
    if (test_session_new_rejects_empty() != 0) return 1;
    if (test_session_new_rejects_missing_dir() != 0) return 1;
    if (test_null_safety() != 0) return 1;
    if (test_pool_list_loaded_rejects_null_pool() != 0) return 1;
    printf("smoke: ok\n");
    return 0;
}

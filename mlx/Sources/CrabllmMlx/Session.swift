// Session.swift — Phase 2 dummy implementation.
//
// Every function declared in `mlx/include/crabllm_mlx.h` has a C entry
// point here via `@_cdecl`. The bodies return canned data so the Rust
// side (Phase 3) can verify the FFI boundary round-trips before we
// pull in mlx-swift-lm. Phase 5 replaces all of this with real model
// loading and inference.
//
// The C header is the source of truth for struct layout. The Swift
// side reads/writes those structs via raw-pointer offsets computed
// below. `tests/smoke.c` holds `_Static_assert`s that pin the offsets
// from the C side — if a future header edit drifts the layout, smoke.c
// fails to compile and this file breaks in lock-step.
//
// Status constants mirror the C header's anonymous enum. They are also
// `_Static_assert`-pinned in smoke.c so a rename on either side is
// caught at build time.

import Foundation

// MARK: - Status constants (pinned by smoke.c _Static_asserts)

private let CRABLLM_MLX_OK: Int32 = 0
private let CRABLLM_MLX_ERR_INVALID_ARG: Int32 = 1
private let CRABLLM_MLX_ERR_UNKNOWN: Int32 = 99

// MARK: - Session type

final class CrabllmMlxSession {
    let modelDir: String

    init(modelDir: String) {
        self.modelDir = modelDir
    }
}

// MARK: - Struct layout (matches crabllm_mlx.h)
//
// CrabllmMlxGenerateRequest (40 bytes on 64-bit):
//   0  : const char *messages_json
//   8  : const char *tools_json
//   16 : CrabllmMlxGenerateOptions options      (24 bytes, see below)
//   40 : const uint32_t *cancel_flag
//
// CrabllmMlxGenerateOptions (24 bytes, 8-aligned):
//   0  : uint64_t seed
//   8  : uint32_t max_tokens
//   12 : float    temperature
//   16 : float    top_p
//   20 : (4 bytes trailing padding so the struct is a multiple of its
//        alignment — required because `Options` is embedded in
//        `Request` between two pointers)
//
// CrabllmMlxGenerateResult (40 bytes on 64-bit):
//   0  : char     *text
//   8  : char     *tool_calls_json
//   16 : uint32_t  prompt_tokens
//   20 : uint32_t  completion_tokens
//   24 : char     *error
//
// Phase 2 only reads `messages_json` (offset 0) from the request. The
// rest of the layout is documented here so Phase 5 has a single place
// to look. Any Phase-5 reader MUST add a matching static assert in
// smoke.c before shipping.

private let resultOffsetText = 0
private let resultOffsetToolCallsJson = 8
private let resultOffsetPromptTokens = 16
private let resultOffsetCompletionTokens = 20
private let resultOffsetError = 24

// MARK: - String helpers

// Copy a Swift String into a fresh C string. Returns nil on allocation
// failure (so library code never traps). Pairs with `free()` in the
// `@_cdecl` release functions.
private func cString(_ s: String) -> UnsafeMutablePointer<CChar>? {
    return s.withCString { strdup($0) }
}

// Convert a borrowed C string to Swift. Returns nil on NULL input.
private func swiftString(_ cStr: UnsafePointer<CChar>?) -> String? {
    guard let cStr = cStr else { return nil }
    return String(cString: cStr)
}

// MARK: - Result struct accessors

private func resultSetText(_ result: UnsafeMutableRawPointer, _ s: String?) {
    let field = result.advanced(by: resultOffsetText)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    field.pointee = s.flatMap { cString($0) }
}

private func resultSetToolCallsJson(_ result: UnsafeMutableRawPointer, _ s: String?) {
    let field = result.advanced(by: resultOffsetToolCallsJson)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    field.pointee = s.flatMap { cString($0) }
}

private func resultSetPromptTokens(_ result: UnsafeMutableRawPointer, _ n: UInt32) {
    result.advanced(by: resultOffsetPromptTokens)
        .assumingMemoryBound(to: UInt32.self)
        .pointee = n
}

private func resultSetCompletionTokens(_ result: UnsafeMutableRawPointer, _ n: UInt32) {
    result.advanced(by: resultOffsetCompletionTokens)
        .assumingMemoryBound(to: UInt32.self)
        .pointee = n
}

private func resultSetError(_ result: UnsafeMutableRawPointer, _ s: String?) {
    let field = result.advanced(by: resultOffsetError)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    field.pointee = s.flatMap { cString($0) }
}

// Reset every field of the result struct to its zero/null value.
// Called at the start of generate*() so partial writes can't leak if
// the caller reused a struct from a prior failed call.
private func resultClear(_ result: UnsafeMutableRawPointer) {
    let textField = result.advanced(by: resultOffsetText)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    textField.pointee = nil
    let toolsField = result.advanced(by: resultOffsetToolCallsJson)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    toolsField.pointee = nil
    let errorField = result.advanced(by: resultOffsetError)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    errorField.pointee = nil
    resultSetPromptTokens(result, 0)
    resultSetCompletionTokens(result, 0)
}

// Free a string field if non-NULL and null the slot so `result_free`
// is idempotent.
private func resultFreeStringField(_ result: UnsafeMutableRawPointer, offset: Int) {
    let field = result.advanced(by: offset)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    if let ptr = field.pointee {
        free(ptr)
        field.pointee = nil
    }
}

// MARK: - Request struct accessors (Phase 2 only reads messages_json)

private func requestMessagesJson(_ request: UnsafeRawPointer) -> String? {
    let field = request.assumingMemoryBound(to: UnsafePointer<CChar>?.self)
    return swiftString(field.pointee)
}

// MARK: - FFI entry points

@_cdecl("crabllm_mlx_session_new")
public func crabllm_mlx_session_new(
    _ modelDirPath: UnsafePointer<CChar>?,
    _ outSession: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let outSession = outSession else {
        if let outError = outError {
            outError.pointee = cString("out_session is NULL")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let path = swiftString(modelDirPath), !path.isEmpty else {
        if let outError = outError {
            outError.pointee = cString("model_dir_path is NULL or empty")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    // Phase 2: accept any non-empty path. Phase 5 will stat the
    // directory and load weights/tokenizer via mlx-swift-lm — at which
    // point smoke.c's "/tmp/dummy-model" stops working and needs a real
    // cached model. See the TODO in tests/smoke.c.
    let session = CrabllmMlxSession(modelDir: path)
    outSession.pointee = Unmanaged.passRetained(session).toOpaque()
    return CRABLLM_MLX_OK
}

@_cdecl("crabllm_mlx_session_free")
public func crabllm_mlx_session_free(_ session: UnsafeMutableRawPointer?) {
    guard let session = session else { return }
    Unmanaged<CrabllmMlxSession>.fromOpaque(session).release()
}

@_cdecl("crabllm_mlx_generate")
public func crabllm_mlx_generate(
    _ session: UnsafeMutableRawPointer?,
    _ request: UnsafeRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else {
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    resultClear(result)

    guard session != nil else {
        resultSetError(result, "session is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let request = request else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let messages = requestMessagesJson(request) ?? "<no messages>"
    let canned = "dummy: echoing " + messages.prefix(64)

    resultSetText(result, canned)
    // Token counts are fake in Phase 2. Use UTF-8 byte length rather
    // than grapheme count so the numbers are at least consistent with
    // the wire encoding. Phase 5 replaces these with the tokenizer's
    // real counts.
    resultSetPromptTokens(result, 4)
    resultSetCompletionTokens(result, UInt32(canned.utf8.count))
    return CRABLLM_MLX_OK
}

@_cdecl("crabllm_mlx_generate_stream")
public func crabllm_mlx_generate_stream(
    _ session: UnsafeMutableRawPointer?,
    _ request: UnsafeRawPointer?,
    _ tokenCb: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32)?,
    _ userData: UnsafeMutableRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else {
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    resultClear(result)

    guard session != nil else {
        resultSetError(result, "session is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard request != nil else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let tokenCb = tokenCb else {
        resultSetError(result, "token_cb is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let tokens = ["hello", " from", " swift", " stub"]
    var completionBytes: UInt32 = 0
    for token in tokens {
        let stop = token.withCString { ptr -> Int32 in
            tokenCb(ptr, userData)
        }
        completionBytes += UInt32(token.utf8.count)
        if stop != 0 { break }
    }

    resultSetPromptTokens(result, 4)
    resultSetCompletionTokens(result, completionBytes)
    return CRABLLM_MLX_OK
}

@_cdecl("crabllm_mlx_result_free")
public func crabllm_mlx_result_free(_ result: UnsafeMutableRawPointer?) {
    guard let result = result else { return }
    resultFreeStringField(result, offset: resultOffsetText)
    resultFreeStringField(result, offset: resultOffsetToolCallsJson)
    resultFreeStringField(result, offset: resultOffsetError)
    resultSetPromptTokens(result, 0)
    resultSetCompletionTokens(result, 0)
}

@_cdecl("crabllm_mlx_string_free")
public func crabllm_mlx_string_free(_ s: UnsafeMutablePointer<CChar>?) {
    guard let s = s else { return }
    free(s)
}

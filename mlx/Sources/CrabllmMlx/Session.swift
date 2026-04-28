// Session.swift — Phase 5 real implementation.
//
// Every function declared in `mlx/include/crabllm_mlx.h` has a C entry
// point here via `@_cdecl`. Phase 2's dummy bodies are replaced with
// calls into `mlx-swift-lm` 2.31.3: `loadModelContainer(directory:)`
// for weights + tokenizer + chat template, and a fresh `ChatSession`
// per `generate*` call driving `streamDetails(to:)` for streaming
// text + tool calls + token counts.
//
// Layout contract: the C structs `CrabllmMlxGenerateRequest` and
// `CrabllmMlxGenerateResult` are accessed via raw-pointer offsets.
// Every offset is pinned by `_Static_assert`s in `tests/smoke.c`;
// edit that file if you change the header.
//
// Threading: the Rust caller wraps every FFI call in
// `tokio::task::spawn_blocking`, so blocking the calling thread on an
// async Task via `DispatchSemaphore` is safe — we are not running on
// a cooperative executor. `@unchecked Sendable` on the session handle
// is justified because `ModelContainer` is already `Sendable` and we
// never mutate the wrapper after construction.

import Foundation
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

// Force the ObjC runtime to load the TrampolineModelFactory classes
// from MLXLLM and MLXVLM. Without these, NSClassFromString lookups
// for "MLXLLM.TrampolineModelFactory" / "MLXVLM.TrampolineModelFactory"
// return nil in a static library because the linker dead-strips the
// unreferenced ObjC classes. `loadModelContainer` auto-dispatches
// through `ModelFactoryRegistry`, which walks both trampolines in
// order (VLM first, then LLM) — if either is missing the corresponding
// model family silently fails to load.
private let _forceLoadLLMFactory: AnyClass? = MLXLLM.TrampolineModelFactory.self
private let _forceLoadVLMFactory: AnyClass? = MLXVLM.TrampolineModelFactory.self

// MARK: - Status constants (pinned by smoke.c _Static_asserts)

let CRABLLM_MLX_OK: Int32 = 0
let CRABLLM_MLX_ERR_INVALID_ARG: Int32 = 1
let CRABLLM_MLX_ERR_MODEL_LOAD: Int32 = 2
let CRABLLM_MLX_ERR_UNSUPPORTED_ARCH: Int32 = 3
let CRABLLM_MLX_ERR_GENERATE: Int32 = 4
let CRABLLM_MLX_ERR_UNKNOWN: Int32 = 99

// MARK: - Session type

final class CrabllmMlxSession: @unchecked Sendable {
    let container: ModelContainer

    init(container: ModelContainer) {
        self.container = container
    }
}

// MARK: - Struct layout (matches crabllm_mlx.h)
//
// CrabllmMlxGenerateRequest (48 bytes on 64-bit):
//   0  : const char *messages_json
//   8  : const char *tools_json
//   16 : CrabllmMlxGenerateOptions options      (24 bytes)
//   40 : const uint32_t *cancel_flag
//
// CrabllmMlxGenerateOptions (24 bytes, 8-aligned):
//   0  : uint64_t seed
//   8  : uint32_t max_tokens
//   12 : float    temperature
//   16 : float    top_p
//   20 : (4 bytes trailing pad)
//
// CrabllmMlxGenerateResult (32 bytes):
//   0  : char     *text
//   8  : char     *tool_calls_json
//   16 : uint32_t  prompt_tokens
//   20 : uint32_t  completion_tokens
//   24 : char     *error

private let requestOffsetMessagesJson = 0
private let requestOffsetToolsJson = 8
private let requestOffsetOptions = 16
private let requestOffsetCancelFlag = 40

private let optionsOffsetSeed = 0
private let optionsOffsetMaxTokens = 8
private let optionsOffsetTemperature = 12
private let optionsOffsetTopP = 16

private let resultOffsetText = 0
private let resultOffsetToolCallsJson = 8
private let resultOffsetPromptTokens = 16
private let resultOffsetCompletionTokens = 20
private let resultOffsetError = 24

// MARK: - String helpers

func cString(_ s: String) -> UnsafeMutablePointer<CChar>? {
    return s.withCString { strdup($0) }
}

func swiftString(_ cStr: UnsafePointer<CChar>?) -> String? {
    guard let cStr = cStr else { return nil }
    return String(cString: cStr)
}

// MARK: - Request struct accessors

struct RequestView {
    let messagesJson: String
    let toolsJson: String?
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let seed: UInt64
    let cancelFlag: UnsafePointer<UInt32>?
}

func parseRequest(_ request: UnsafeRawPointer) -> RequestView? {
    let messagesPtrField = request.advanced(by: requestOffsetMessagesJson)
        .assumingMemoryBound(to: UnsafePointer<CChar>?.self)
    guard let messages = swiftString(messagesPtrField.pointee), !messages.isEmpty else {
        return nil
    }

    let toolsPtrField = request.advanced(by: requestOffsetToolsJson)
        .assumingMemoryBound(to: UnsafePointer<CChar>?.self)
    let tools = swiftString(toolsPtrField.pointee)

    let optionsBase = request.advanced(by: requestOffsetOptions)
    let seed = optionsBase.advanced(by: optionsOffsetSeed)
        .assumingMemoryBound(to: UInt64.self).pointee
    let maxTokensRaw = optionsBase.advanced(by: optionsOffsetMaxTokens)
        .assumingMemoryBound(to: UInt32.self).pointee
    let temperatureRaw = optionsBase.advanced(by: optionsOffsetTemperature)
        .assumingMemoryBound(to: Float.self).pointee
    let topPRaw = optionsBase.advanced(by: optionsOffsetTopP)
        .assumingMemoryBound(to: Float.self).pointee

    let cancelField = request.advanced(by: requestOffsetCancelFlag)
        .assumingMemoryBound(to: UnsafePointer<UInt32>?.self)
    let cancelFlag = cancelField.pointee

    return RequestView(
        messagesJson: messages,
        toolsJson: tools,
        maxTokens: maxTokensRaw == 0 ? nil : Int(maxTokensRaw),
        temperature: temperatureRaw > 0 ? temperatureRaw : nil,
        topP: topPRaw > 0 ? topPRaw : nil,
        seed: seed,
        cancelFlag: cancelFlag
    )
}

// MARK: - Result struct accessors

func resultSetText(_ result: UnsafeMutableRawPointer, _ s: String?) {
    let field = result.advanced(by: resultOffsetText)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    field.pointee = s.flatMap { cString($0) }
}

func resultSetToolCallsJson(_ result: UnsafeMutableRawPointer, _ s: String?) {
    let field = result.advanced(by: resultOffsetToolCallsJson)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    field.pointee = s.flatMap { cString($0) }
}

func resultSetPromptTokens(_ result: UnsafeMutableRawPointer, _ n: UInt32) {
    result.advanced(by: resultOffsetPromptTokens)
        .assumingMemoryBound(to: UInt32.self).pointee = n
}

func resultSetCompletionTokens(_ result: UnsafeMutableRawPointer, _ n: UInt32) {
    result.advanced(by: resultOffsetCompletionTokens)
        .assumingMemoryBound(to: UInt32.self).pointee = n
}

func resultSetError(_ result: UnsafeMutableRawPointer, _ s: String?) {
    let field = result.advanced(by: resultOffsetError)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    field.pointee = s.flatMap { cString($0) }
}

func resultClear(_ result: UnsafeMutableRawPointer) {
    resultSetText(result, nil)
    resultSetToolCallsJson(result, nil)
    resultSetError(result, nil)
    resultSetPromptTokens(result, 0)
    resultSetCompletionTokens(result, 0)
}

func resultFreeStringField(_ result: UnsafeMutableRawPointer, offset: Int) {
    let field = result.advanced(by: offset)
        .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
    if let ptr = field.pointee {
        free(ptr)
        field.pointee = nil
    }
}

// MARK: - Sync-from-async bridge

/// Run an async block synchronously by spinning a detached Task and
/// blocking on a DispatchSemaphore.
///
/// **Caller contract:** this function blocks the calling thread and
/// MUST NOT be called from a thread that participates in the Swift
/// cooperative executor. Every Rust caller wraps the FFI call in
/// `tokio::task::spawn_blocking`, so the calling thread is a dedicated
/// OS worker, not a tokio runtime worker and not a Swift async task.
/// Violating this will deadlock if `op` awaits work scheduled onto
/// the same executor.
///
/// **Memory ordering:** the detached Task writes `result` and then
/// signals the semaphore; the caller `wait`s and reads `result`.
/// `DispatchSemaphore.signal` / `wait` provide release/acquire pairing
/// on Darwin, so the write is visible to the reader.
/// `nonisolated(unsafe)` is needed because `Result<T, Error>` is not
/// `Sendable` for arbitrary `T`; the pointer hand-off is single-shot
/// and mediated by the semaphore.
func blockingAwait<T>(_ op: @Sendable @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<T, Error>?
    Task.detached {
        do {
            let value = try await op()
            result = .success(value)
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    switch result! {
    case .success(let v): return v
    case .failure(let e): throw e
    }
}

// MARK: - Internal error type

enum FFIError: Error {
    case invalidArg(String)
    case modelLoad(String)
    case unsupportedArch(String)
    case generate(String)

    var status: Int32 {
        switch self {
        case .invalidArg: return CRABLLM_MLX_ERR_INVALID_ARG
        case .modelLoad: return CRABLLM_MLX_ERR_MODEL_LOAD
        case .unsupportedArch: return CRABLLM_MLX_ERR_UNSUPPORTED_ARCH
        case .generate: return CRABLLM_MLX_ERR_GENERATE
        }
    }

    var message: String {
        switch self {
        case .invalidArg(let s), .modelLoad(let s), .unsupportedArch(let s), .generate(let s):
            return s
        }
    }
}

func translateModelLoadError(_ error: Error) -> FFIError {
    // `ModelFactoryError.unsupportedModelType(String)` is the signal we
    // care about — surface it with a distinct status so the Rust side
    // can return OpenAI `model_not_found`. Every other variant maps to
    // the generic MODEL_LOAD status. `ModelFactoryError` is not marked
    // `@frozen` in mlx-swift-lm, but it is a same-module enum so
    // `@unknown default` is rejected by the compiler — we take the
    // exhaustive-default route and accept that a future variant will
    // land here silently. Revisit after the next mlx-swift-lm bump.
    if let factoryErr = error as? ModelFactoryError {
        switch factoryErr {
        case .unsupportedModelType(let arch):
            return .unsupportedArch("unsupported architecture: \(arch)")
        default:
            return .modelLoad("\(factoryErr)")
        }
    }
    return .modelLoad("\(error)")
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

    let url = URL(fileURLWithPath: path)
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
        if let outError = outError {
            outError.pointee = cString("model_dir_path does not exist or is not a directory: \(path)")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    do {
        let container = try blockingAwait {
            try await loadModelContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
        }
        let session = CrabllmMlxSession(container: container)
        outSession.pointee = Unmanaged.passRetained(session).toOpaque()
        return CRABLLM_MLX_OK
    } catch {
        let ffi = translateModelLoadError(error)
        if let outError = outError {
            outError.pointee = cString(ffi.message)
        }
        return ffi.status
    }
}

@_cdecl("crabllm_mlx_session_free")
public func crabllm_mlx_session_free(_ session: UnsafeMutableRawPointer?) {
    guard let session = session else { return }
    Unmanaged<CrabllmMlxSession>.fromOpaque(session).release()
}

/// Shared code path for both generate and generate_stream. The
/// `onChunk` closure is nil for non-streaming and swallows each text
/// chunk for streaming. Returns the final accumulated text (always
/// populated; the streaming caller discards it).
/// Core generation driving a `ModelContainer`. Called by both the
/// session FFI and the pool FFI so the logic lives in one place.
func runGenerationWithContainer(
    _ container: ModelContainer,
    _ view: RequestView,
    onChunk: ((String) -> Bool)?
) throws -> (text: String, toolCallsJson: String?, promptTokens: UInt32, completionTokens: UInt32) {
    let decoded: DecodedMessages
    do {
        decoded = try decodeMessages(view.messagesJson)
    } catch let e as FFIError {
        throw e
    }

    let tools: [ToolSpec]?
    do {
        tools = try decodeTools(view.toolsJson)
    } catch let e as FFIError {
        throw e
    }

    // Build generation parameters. mlx-swift-lm 2.31.3's
    // GenerateParameters has no seed field — sampling is always
    // randomised. The Rust caller's seed is currently ignored; if a
    // future mlx-swift-lm exposes a seed knob, plumb it here.
    _ = view.seed
    var params = GenerateParameters()
    if let t = view.temperature { params.temperature = t }
    if let p = view.topP { params.topP = p }
    if let m = view.maxTokens { params.maxTokens = m }

    return try blockingAwait { [container, params, tools, decoded] in
        let chat = ChatSession(
            container,
            instructions: decoded.instructions,
            history: decoded.history,
            generateParameters: params,
            tools: tools,
            toolDispatch: nil  // surface tool calls to the caller instead of auto-dispatching
        )

        var accumulated = ""
        var toolCalls: [ToolCall] = []
        var promptTokens: UInt32 = 0
        var completionTokens: UInt32 = 0
        var stoppedByCaller = false

        let stream = chat.streamDetails(
            to: decoded.lastPrompt,
            role: decoded.lastRole,
            images: decoded.lastImages,
            videos: []
        )
        for try await gen in stream {
            // Honor Rust-side cancellation between chunks. The load is
            // a plain `pointee` rather than a C11 acquire load; on
            // ARM64 Darwin this is functionally acquire-ordered for
            // an aligned 4-byte read because:
            //   * the Rust writer is contractually obligated to use
            //     `Ordering::Release` or stronger (see the module
            //     header of `crates/mlx/src/session.rs`);
            //   * each `try await` suspension point is an effective
            //     compiler barrier, so the read cannot be hoisted
            //     out of the loop;
            //   * the Swift concurrency runtime performs atomic
            //     operations across resume boundaries, which act as
            //     synchronization points for the enclosing thread.
            // If a future port needs strict C11 atomics, wrap the
            // pointer with swift-atomics' `UnsafeAtomic` or vend a
            // small C helper that issues `atomic_load_explicit`.
            if let flag = view.cancelFlag, flag.pointee != 0 {
                break
            }
            switch gen {
            case .chunk(let s):
                accumulated.append(s)
                if let onChunk = onChunk, onChunk(s) {
                    stoppedByCaller = true
                }
            case .info(let info):
                promptTokens = UInt32(max(0, info.promptTokenCount))
                completionTokens = UInt32(max(0, info.generationTokenCount))
            case .toolCall(let call):
                toolCalls.append(call)
            }
            if stoppedByCaller { break }
        }

        // Belt-and-suspenders: a non-empty user prompt must produce a
        // non-zero prompt token count. If we got here with zero, the
        // chat-template render likely silently dropped the input
        // (e.g. an unsupported feature like tools or system messages
        // on a template that doesn't slot them in). Surface it as an
        // explicit error rather than returning success-with-zero-output.
        if !stoppedByCaller && !decoded.lastPrompt.isEmpty && promptTokens == 0 {
            throw FFIError.generate(
                "model produced zero prompt tokens despite non-empty input — "
                + "likely an unsupported chat-template feature (tools, system message, etc.)"
            )
        }

        let toolCallsJson: String?
        if toolCalls.isEmpty {
            toolCallsJson = nil
        } else {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            do {
                let data = try encoder.encode(toolCalls)
                toolCallsJson = String(data: data, encoding: .utf8)
            } catch {
                throw FFIError.generate("failed to encode tool calls: \(error)")
            }
        }

        return (accumulated, toolCallsJson, promptTokens, completionTokens)
    }
}

private func withSession(
    _ session: UnsafeMutableRawPointer?,
    _ body: (CrabllmMlxSession) throws -> Int32
) rethrows -> Int32 {
    guard let session = session else {
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    let unmanaged = Unmanaged<CrabllmMlxSession>.fromOpaque(session).takeUnretainedValue()
    return try body(unmanaged)
}

@_cdecl("crabllm_mlx_generate")
public func crabllm_mlx_generate(
    _ session: UnsafeMutableRawPointer?,
    _ request: UnsafeRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else { return CRABLLM_MLX_ERR_INVALID_ARG }
    resultClear(result)

    guard let request = request else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let view = parseRequest(request) else {
        resultSetError(result, "messages_json is NULL, empty, or not valid UTF-8")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    return withSession(session) { session in
        do {
            let out = try runGenerationWithContainer(session.container, view, onChunk: nil)
            resultSetText(result, out.text)
            resultSetToolCallsJson(result, out.toolCallsJson)
            resultSetPromptTokens(result, out.promptTokens)
            resultSetCompletionTokens(result, out.completionTokens)
            return CRABLLM_MLX_OK
        } catch let e as FFIError {
            resultSetError(result, e.message)
            return e.status
        } catch {
            resultSetError(result, "unexpected generate error: \(error)")
            return CRABLLM_MLX_ERR_UNKNOWN
        }
    }
}

@_cdecl("crabllm_mlx_generate_stream")
public func crabllm_mlx_generate_stream(
    _ session: UnsafeMutableRawPointer?,
    _ request: UnsafeRawPointer?,
    _ tokenCb: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32)?,
    _ userData: UnsafeMutableRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else { return CRABLLM_MLX_ERR_INVALID_ARG }
    resultClear(result)

    guard let request = request else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let view = parseRequest(request) else {
        resultSetError(result, "messages_json is NULL, empty, or not valid UTF-8")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let tokenCb = tokenCb else {
        resultSetError(result, "token_cb is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    return withSession(session) { session in
        do {
            let out = try runGenerationWithContainer(session.container, view, onChunk: { chunk in
                let stop = chunk.withCString { ptr -> Int32 in
                    tokenCb(ptr, userData)
                }
                return stop != 0
            })
            // Streaming contract: text is always NULL — the caller
            // reconstructed it from the callback stream.
            resultSetToolCallsJson(result, out.toolCallsJson)
            resultSetPromptTokens(result, out.promptTokens)
            resultSetCompletionTokens(result, out.completionTokens)
            _ = out.text
            return CRABLLM_MLX_OK
        } catch let e as FFIError {
            resultSetError(result, e.message)
            return e.status
        } catch {
            resultSetError(result, "unexpected generate_stream error: \(error)")
            return CRABLLM_MLX_ERR_UNKNOWN
        }
    }
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

// Pool.swift — Swift-side multi-model pool with idle eviction.
//
// The pool owns `ModelContainer` instances keyed by local directory
// path. Models are loaded on first request (via `loadModelContainer`)
// and evicted after the idle timeout. An internal `Task` wakes every
// 60 seconds to sweep expired entries.
//
// The pool is an actor so all mutations are serialized by Swift's
// concurrency runtime. FFI entry points bridge from the blocking C
// ABI via `blockingAwait` — same contract as the session functions.

import Foundation
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

private let DEFAULT_IDLE_TIMEOUT: TimeInterval = 30 * 60

// MARK: - Pool actor

actor MlxPool {
    private var models: [String: Entry] = [:]
    private let idleTimeout: TimeInterval
    private var monitorTask: Task<Void, Never>?

    struct Entry {
        let container: ModelContainer
        var lastUsed: Date
    }

    init(idleTimeout: TimeInterval) {
        self.idleTimeout = idleTimeout > 0 ? idleTimeout : DEFAULT_IDLE_TIMEOUT
    }

    func start() {
        monitorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                await self?.evictExpired()
            }
        }
    }

    func ensureLoaded(_ modelDir: String) async throws -> ModelContainer {
        if var entry = models[modelDir] {
            entry.lastUsed = Date()
            models[modelDir] = entry
            return entry.container
        }

        let url = URL(fileURLWithPath: modelDir)
        let container = try await loadModelContainer(
            from: url,
            using: #huggingFaceTokenizerLoader()
        )
        models[modelDir] = Entry(container: container, lastUsed: Date())
        return container
    }

    func evict(_ modelDir: String) {
        models.removeValue(forKey: modelDir)
    }

    func stopAll() {
        models.removeAll()
        monitorTask?.cancel()
        monitorTask = nil
    }

    /// Snapshot every loaded slot's name and last-used time. Returns
    /// the minimum the FFI wrapper needs to build a `LoadedModel`
    /// array; memory-footprint computation is a filesystem scan and
    /// runs *outside* the actor so it doesn't block concurrent
    /// generate / evict calls.
    func snapshot() -> [(name: String, lastUsedUnix: Int64)] {
        models.map { (dir, entry) in
            (name: dir, lastUsedUnix: Int64(entry.lastUsed.timeIntervalSince1970))
        }
    }

    private func evictExpired() {
        let now = Date()
        let expired = models.filter { now.timeIntervalSince($0.value.lastUsed) > idleTimeout }
        for key in expired.keys {
            models.removeValue(forKey: key)
        }
    }

    deinit {
        monitorTask?.cancel()
    }
}

// MARK: - FFI wrappers

// The pool handle is an `Unmanaged<MlxPoolBox>` because actors cannot
// be directly retained via `Unmanaged`. We box the actor in a plain
// class and manage that.
private final class MlxPoolBox: @unchecked Sendable {
    let pool: MlxPool
    init(_ pool: MlxPool) { self.pool = pool }
}

@_cdecl("crabllm_mlx_pool_new")
public func crabllm_mlx_pool_new(
    _ idleTimeoutSecs: UInt64,
    _ outPool: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let outPool = outPool else {
        if let outError = outError {
            outError.pointee = cString("out_pool is NULL")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let pool = MlxPool(idleTimeout: TimeInterval(idleTimeoutSecs))

    // Start the idle monitor via blockingAwait since actor methods are
    // async. `start()` is fast (just spawns a Task), so this is
    // effectively instant.
    do {
        try blockingAwait { await pool.start() }
    } catch {
        if let outError = outError {
            outError.pointee = cString("pool start failed: \(error)")
        }
        return CRABLLM_MLX_ERR_UNKNOWN
    }

    let box = MlxPoolBox(pool)
    outPool.pointee = Unmanaged.passRetained(box).toOpaque()
    return CRABLLM_MLX_OK
}

@_cdecl("crabllm_mlx_pool_free")
public func crabllm_mlx_pool_free(_ pool: UnsafeMutableRawPointer?) {
    guard let pool = pool else { return }
    let box = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeRetainedValue()
    // Best-effort stop. If the caller forgot to call pool_stop_all,
    // at least cancel the monitor task so it doesn't leak.
    _ = box  // ARC releases the box, which deinits the actor, which cancels the monitor.
}

@_cdecl("crabllm_mlx_pool_generate")
public func crabllm_mlx_pool_generate(
    _ pool: UnsafeMutableRawPointer?,
    _ modelDirPath: UnsafePointer<CChar>?,
    _ request: UnsafeRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else { return CRABLLM_MLX_ERR_INVALID_ARG }
    resultClear(result)

    guard let pool = pool else {
        resultSetError(result, "pool is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let modelDir = swiftString(modelDirPath), !modelDir.isEmpty else {
        resultSetError(result, "model_dir_path is NULL or empty")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let request = request else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let view = parseRequest(request) else {
        resultSetError(result, "messages_json is NULL, empty, or not valid UTF-8")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    do {
        let container = try blockingAwait { try await actor.ensureLoaded(modelDir) }
        let out = try runGenerationWithContainer(container, view, onChunk: nil)
        resultSetText(result, out.text)
        resultSetToolCallsJson(result, out.toolCallsJson)
        resultSetPromptTokens(result, out.promptTokens)
        resultSetCompletionTokens(result, out.completionTokens)
        return CRABLLM_MLX_OK
    } catch let e as FFIError {
        resultSetError(result, e.message)
        return e.status
    } catch {
        resultSetError(result, "pool generate error: \(error)")
        return CRABLLM_MLX_ERR_UNKNOWN
    }
}

@_cdecl("crabllm_mlx_pool_generate_stream")
public func crabllm_mlx_pool_generate_stream(
    _ pool: UnsafeMutableRawPointer?,
    _ modelDirPath: UnsafePointer<CChar>?,
    _ request: UnsafeRawPointer?,
    _ tokenCb: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32)?,
    _ userData: UnsafeMutableRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else { return CRABLLM_MLX_ERR_INVALID_ARG }
    resultClear(result)

    guard let pool = pool else {
        resultSetError(result, "pool is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let modelDir = swiftString(modelDirPath), !modelDir.isEmpty else {
        resultSetError(result, "model_dir_path is NULL or empty")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
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

    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    do {
        let container = try blockingAwait { try await actor.ensureLoaded(modelDir) }
        let out = try runGenerationWithContainer(container, view, onChunk: { chunk in
            let stop = chunk.withCString { ptr -> Int32 in
                tokenCb(ptr, userData)
            }
            return stop != 0
        })
        resultSetToolCallsJson(result, out.toolCallsJson)
        resultSetPromptTokens(result, out.promptTokens)
        resultSetCompletionTokens(result, out.completionTokens)
        _ = out.text
        return CRABLLM_MLX_OK
    } catch let e as FFIError {
        resultSetError(result, e.message)
        return e.status
    } catch {
        resultSetError(result, "pool generate_stream error: \(error)")
        return CRABLLM_MLX_ERR_UNKNOWN
    }
}

@_cdecl("crabllm_mlx_pool_evict")
public func crabllm_mlx_pool_evict(
    _ pool: UnsafeMutableRawPointer?,
    _ modelDirPath: UnsafePointer<CChar>?
) {
    guard let pool = pool, let dir = swiftString(modelDirPath) else { return }
    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    try? blockingAwait { await actor.evict(dir) }
}

@_cdecl("crabllm_mlx_pool_stop_all")
public func crabllm_mlx_pool_stop_all(_ pool: UnsafeMutableRawPointer?) {
    guard let pool = pool else { return }
    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    try? blockingAwait { await actor.stopAll() }
}

// MARK: - Loaded model inventory
//
// The C ABI struct `CrabllmMlxLoadedModel` is written via byte-offset
// stores against an `UnsafeMutableRawPointer` buffer, mirroring the
// `CrabllmMlxGenerateResult` pattern in Session.swift. We don't define
// a Swift struct for it because plain Swift structs have no layout
// guarantee — smoke.c pins the C-side layout, and these offset
// constants are the Swift-side mirror pinned to those same values.

private let loadedOffsetName = 0
private let loadedOffsetMemoryBytes = 8
private let loadedOffsetLastUsedUnix = 16
private let loadedModelStride = 24

/// Sum of weight-file sizes under a model directory. Best-effort:
/// unreadable / missing paths contribute zero. Runs outside the actor
/// so concurrent generate/evict aren't blocked on filesystem I/O.
private func bestEffortWeightBytes(atPath dir: String) -> UInt64 {
    let fm = FileManager.default
    let url = URL(fileURLWithPath: dir)
    guard let entries = try? fm.contentsOfDirectory(
        at: url,
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles]
    ) else {
        return 0
    }
    var total: UInt64 = 0
    for entry in entries {
        let ext = entry.pathExtension.lowercased()
        guard ext == "safetensors" || ext == "bin" || ext == "gguf" else { continue }
        if let size = try? entry.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            total &+= UInt64(size)
        }
    }
    return total
}

/// Free all strdup'd name pointers in a loaded-model buffer, then
/// deallocate the buffer. `count` must match the allocation.
private func loadedBufferFreeNames(_ buf: UnsafeMutableRawPointer, _ count: Int) {
    for i in 0..<count {
        let slot = buf.advanced(by: i * loadedModelStride + loadedOffsetName)
            .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
        if let name = slot.pointee {
            free(UnsafeMutableRawPointer(name))
            slot.pointee = nil
        }
    }
}

@_cdecl("crabllm_mlx_pool_list_loaded")
public func crabllm_mlx_pool_list_loaded(
    _ pool: UnsafeMutableRawPointer?,
    _ outArray: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outCount: UnsafeMutablePointer<Int>?,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let outArray = outArray, let outCount = outCount else {
        if let outError = outError {
            outError.pointee = cString("out_array / out_count is NULL")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    outArray.pointee = nil
    outCount.pointee = 0

    guard let pool = pool else {
        if let outError = outError {
            outError.pointee = cString("pool is NULL")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    let snapshot: [(name: String, lastUsedUnix: Int64)]
    do {
        snapshot = try blockingAwait { await actor.snapshot() }
    } catch {
        if let outError = outError {
            outError.pointee = cString("pool list_loaded error: \(error)")
        }
        return CRABLLM_MLX_ERR_UNKNOWN
    }

    let count = snapshot.count
    if count == 0 {
        return CRABLLM_MLX_OK
    }

    // Compute memory footprints outside the actor — each one is a
    // FileManager scan, and we don't want to serialize all pool
    // operations behind a directory enumeration.
    let memoryBytes: [UInt64] = snapshot.map { bestEffortWeightBytes(atPath: $0.name) }

    // Raw-byte buffer; field layout is pinned by smoke.c's
    // _Static_assert on the C side and the `loadedOffset*` constants
    // on this side.
    let buf = UnsafeMutableRawPointer.allocate(
        byteCount: count * loadedModelStride,
        alignment: MemoryLayout<UInt64>.alignment
    )
    // Zero the buffer so partial-failure cleanup sees NULL name slots.
    buf.initializeMemory(as: UInt8.self, repeating: 0, count: count * loadedModelStride)

    for (idx, entry) in snapshot.enumerated() {
        guard let namePtr = cString(entry.name) else {
            // strdup OOM: release any names we've already written,
            // deallocate, and report failure.
            loadedBufferFreeNames(buf, idx)
            buf.deallocate()
            if let outError = outError {
                outError.pointee = cString("pool list_loaded: out of memory")
            }
            return CRABLLM_MLX_ERR_UNKNOWN
        }
        let base = buf.advanced(by: idx * loadedModelStride)
        base.advanced(by: loadedOffsetName)
            .assumingMemoryBound(to: UnsafeMutablePointer<CChar>?.self)
            .pointee = namePtr
        base.advanced(by: loadedOffsetMemoryBytes)
            .assumingMemoryBound(to: UInt.self)
            .pointee = UInt(memoryBytes[idx])
        base.advanced(by: loadedOffsetLastUsedUnix)
            .assumingMemoryBound(to: Int64.self)
            .pointee = entry.lastUsedUnix
    }
    outArray.pointee = buf
    outCount.pointee = count
    return CRABLLM_MLX_OK
}

@_cdecl("crabllm_mlx_pool_loaded_free")
public func crabllm_mlx_pool_loaded_free(
    _ array: UnsafeMutableRawPointer?,
    _ count: Int
) {
    guard let array = array else { return }
    loadedBufferFreeNames(array, count)
    array.deallocate()
}

// MessageParsing.swift — decode the OpenAI-shape `messages` + `tools`
// JSON that crosses the Rust/Swift FFI boundary into the Swift types
// mlx-swift-lm's `ChatSession` expects.
//
// Everything in this file is pure data transformation plus the
// synchronous image-fetch helper used for `http(s)://` image URLs.
// No FFI pointers, no C entry points — those stay in `Session.swift`.
// This file exists so `Session.swift` doesn't grow past the point
// where it's painful to open.

import CoreImage
import Foundation
import MLXLMCommon
import Speech

// MARK: - Message parsing

/// Decode the Rust-supplied messages JSON into the shape `ChatSession`
/// expects: a system prompt (if first message is `system`), a history
/// of prior turns, and the final user prompt to respond to.
///
/// Content handling:
///   * Plain string content (`"content": "..."`) is used as-is.
///   * Array-of-parts content (`"content": [{"type":"text","text":"..."},
///     {"type":"image_url","image_url":{"url":"..."}},
///     {"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}]`)
///     is walked: text parts concatenated into `content`, image parts
///     decoded into `UserInput.Image` values and attached to the
///     message, audio parts transcribed via `SFSpeechRecognizer` and
///     appended to the text buffer. Other unknown part types are dropped.
///   * Missing / null / non-string / non-array content becomes "".
///
/// Image URLs in `image_url` parts may be any of:
///   * `data:image/*;base64,...` — base64-decoded inline.
///   * `http(s)://...` — synchronously fetched (safe: the FFI call runs
///     on a dedicated tokio `spawn_blocking` worker, not a cooperative
///     executor).
///   * `file:///...` — passed through as `.url(URL)` for lazy load.
///
/// Known limitations:
///   * `tool_call_id` is dropped. mlx-swift-lm's `Chat.Message.tool(_:)`
///     takes only content; the tokenizer chat template is expected to
///     do the right thing based on ordering alone. Images on `tool`
///     messages are dropped for the same reason — `.tool(_:)` has no
///     `images:` parameter.
///   * Only the first `system` message populates `instructions`; any
///     subsequent `system` messages are added to `history` as
///     `.system(...)` and whether they take effect depends on the
///     model's chat template.
///   * Images on the extracted system prompt are dropped —
///     `ChatSession.instructions` is a plain `String`, not a
///     `Chat.Message`.
struct DecodedMessages {
    let instructions: String?
    let history: [Chat.Message]
    let lastPrompt: String
    let lastImages: [UserInput.Image]
    let lastRole: Chat.Message.Role
}

func decodeMessages(_ json: String) throws -> DecodedMessages {
    guard let data = json.data(using: .utf8) else {
        throw FFIError.invalidArg("messages_json not valid UTF-8")
    }
    guard let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
        throw FFIError.invalidArg("messages_json must be a JSON array of objects")
    }
    if arr.isEmpty {
        throw FFIError.invalidArg("messages_json is empty")
    }

    var messages = arr
    var instructions: String? = nil
    if messages.first?["role"] as? String == "system" {
        let parsed = try parseContent(messages.removeFirst()["content"])
        // `ChatSession.instructions` is `String?`, so collapse empty
        // system prompts to nil. Non-system paths take the empty
        // string verbatim because `streamDetails(to:)` takes `String`.
        instructions = parsed.text.isEmpty ? nil : parsed.text
    }

    guard let last = messages.last else {
        throw FFIError.invalidArg("messages_json has no non-system messages")
    }
    let lastRole: Chat.Message.Role = switch last["role"] as? String {
    case "user": .user
    case "assistant": .assistant
    case "tool": .tool
    case "system": .system
    default: .user
    }
    let lastParsed = try parseContent(last["content"])

    var history: [Chat.Message] = []
    for msg in messages.dropLast() {
        let parsed = try parseContent(msg["content"])
        switch msg["role"] as? String {
        case "assistant":
            history.append(.assistant(parsed.text, images: parsed.images))
        case "system":
            history.append(.system(parsed.text, images: parsed.images))
        case "tool":
            history.append(.tool(parsed.text))
        default:
            history.append(.user(parsed.text, images: parsed.images))
        }
    }

    return DecodedMessages(
        instructions: instructions,
        history: history,
        lastPrompt: lastParsed.text,
        lastImages: lastParsed.images,
        lastRole: lastRole
    )
}

/// Walk an OpenAI-shape `content` payload and extract text + images.
///
/// Plain string content round-trips as `(content, [])`. Array-of-parts
/// content is split: `text` parts accumulate into the text buffer,
/// `image_url` parts are decoded into `UserInput.Image`. Any other part
/// type is dropped. NULL / unrecognized input yields `("", [])`.
func parseContent(_ value: Any?) throws -> (text: String, images: [UserInput.Image]) {
    guard let value = value else { return ("", []) }
    if let s = value as? String {
        return (s, [])
    }
    guard let parts = value as? [[String: Any]] else {
        return ("", [])
    }
    var text = ""
    var images: [UserInput.Image] = []
    for part in parts {
        switch part["type"] as? String {
        case "text":
            if let t = part["text"] as? String { text.append(t) }
        case "image_url":
            guard let urlObj = part["image_url"] as? [String: Any],
                  let urlStr = urlObj["url"] as? String, !urlStr.isEmpty
            else {
                throw FFIError.invalidArg("image_url part is missing url")
            }
            images.append(try decodeImageURL(urlStr))
        case "input_audio":
            guard let audioObj = part["input_audio"] as? [String: Any],
                  let b64 = audioObj["data"] as? String, !b64.isEmpty
            else {
                throw FFIError.invalidArg("input_audio part is missing data")
            }
            let format = (audioObj["format"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "wav"
            text.append(try transcribeAudio(data: b64, format: format))
        default:
            continue
        }
    }
    return (text, images)
}

/// Decode an OpenAI `image_url.url` into a `UserInput.Image`.
/// Supports inline `data:` URIs, remote `http(s)://`, and local
/// `file://` schemes.
func decodeImageURL(_ urlStr: String) throws -> UserInput.Image {
    if urlStr.hasPrefix("data:") {
        return try decodeDataURL(urlStr)
    }
    guard let url = URL(string: urlStr) else {
        throw FFIError.invalidArg("image_url.url is not a valid URL")
    }
    switch url.scheme?.lowercased() {
    case "http", "https":
        let data = try fetchImageBytes(url)
        guard let image = CIImage(data: data) else {
            throw FFIError.invalidArg("image_url payload is not a decodable image")
        }
        return .ciImage(image)
    case "file":
        return .url(url)
    default:
        throw FFIError.invalidArg("image_url scheme must be data, http(s), or file")
    }
}

/// Synchronously fetch bytes from an `http(s)` URL with a bounded
/// timeout. `URLSession` enforces per-request and per-resource
/// deadlines on its own, and the outer semaphore wait guards against
/// any Darwin-internal hang by timing out one second past the resource
/// deadline — a hung server cannot pin the calling worker forever.
/// `.ephemeral` avoids caching fetched images in memory.
private let imageFetchRequestTimeout: TimeInterval = 15
private let imageFetchResourceTimeout: TimeInterval = 30

func fetchImageBytes(_ url: URL) throws -> Data {
    let config = URLSessionConfiguration.ephemeral
    config.timeoutIntervalForRequest = imageFetchRequestTimeout
    config.timeoutIntervalForResource = imageFetchResourceTimeout
    let session = URLSession(configuration: config)
    defer { session.finishTasksAndInvalidate() }

    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var resultData: Data?
    nonisolated(unsafe) var resultError: Error?
    nonisolated(unsafe) var statusCode: Int = 0

    let task = session.dataTask(with: url) { data, response, error in
        resultError = error
        resultData = data
        if let http = response as? HTTPURLResponse {
            statusCode = http.statusCode
        }
        semaphore.signal()
    }
    task.resume()

    let waitDeadline = DispatchTime.now() + imageFetchResourceTimeout + 1
    if semaphore.wait(timeout: waitDeadline) == .timedOut {
        task.cancel()
        throw FFIError.invalidArg("image_url fetch exceeded resource timeout")
    }
    if let error = resultError {
        throw FFIError.invalidArg("image_url fetch failed: \(error)")
    }
    if statusCode != 0 && !(200...299).contains(statusCode) {
        throw FFIError.invalidArg("image_url fetch returned HTTP \(statusCode)")
    }
    guard let data = resultData else {
        throw FFIError.invalidArg("image_url fetch returned no data")
    }
    return data
}

/// Transcribe base64-encoded audio via Apple's `SFSpeechRecognizer`.
///
/// Same threading contract as `fetchImageBytes` — the FFI call runs
/// on a dedicated `spawn_blocking` worker, so blocking on a
/// `DispatchSemaphore` is safe. The recognizer runs on CPU/ANE,
/// leaving the GPU free for MLX inference.
///
/// The `format` parameter becomes the temp file extension so
/// `SFSpeechURLRecognitionRequest` can detect the audio codec
/// (wav, mp3, m4a, etc.). Authorization is requested on first use;
/// if the user denies it, subsequent calls fail immediately.
private let audioTranscriptionTimeout: TimeInterval = 60

func transcribeAudio(data b64: String, format: String) throws -> String {
    // Request authorization if not yet determined.
    let status = SFSpeechRecognizer.authorizationStatus()
    if status == .notDetermined {
        let authSemaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var granted = false
        SFSpeechRecognizer.requestAuthorization { s in
            granted = (s == .authorized)
            authSemaphore.signal()
        }
        if authSemaphore.wait(timeout: .now() + 120) == .timedOut {
            throw FFIError.invalidArg(
                "input_audio: speech recognition authorization prompt timed out"
            )
        }
        if !granted {
            throw FFIError.invalidArg(
                "input_audio: speech recognition authorization denied — "
                + "grant permission in System Settings → Privacy & Security → Speech Recognition"
            )
        }
    } else if status != .authorized {
        throw FFIError.invalidArg(
            "input_audio: speech recognition not authorized (status \(status.rawValue)) — "
            + "grant permission in System Settings → Privacy & Security → Speech Recognition"
        )
    }

    guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable else {
        throw FFIError.invalidArg("input_audio: SFSpeechRecognizer is not available")
    }
    guard let audioData = Data(base64Encoded: b64) else {
        throw FFIError.invalidArg("input_audio: base64 payload failed to decode")
    }

    // Write to a temp file — SFSpeechURLRecognitionRequest needs a file URL.
    let tempFile = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension(format)
    try audioData.write(to: tempFile)
    defer { try? FileManager.default.removeItem(at: tempFile) }

    let request = SFSpeechURLRecognitionRequest(url: tempFile)
    request.shouldReportPartialResults = false

    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var transcript: String?
    nonisolated(unsafe) var recognitionError: Error?

    let task = recognizer.recognitionTask(with: request) { result, error in
        if let error = error {
            recognitionError = error
            semaphore.signal()
            return
        }
        if let result = result, result.isFinal {
            transcript = result.bestTranscription.formattedString
            semaphore.signal()
        }
    }

    let waitDeadline = DispatchTime.now() + audioTranscriptionTimeout
    if semaphore.wait(timeout: waitDeadline) == .timedOut {
        task.cancel()
        throw FFIError.invalidArg("input_audio: transcription exceeded \(Int(audioTranscriptionTimeout))s timeout")
    }
    if let error = recognitionError {
        throw FFIError.invalidArg("input_audio: transcription failed: \(error)")
    }
    return transcript ?? ""
}

/// Decode a `data:[<mediatype>];base64,<payload>` URL. Anchors on the
/// `;base64,` sentinel rather than the first comma so mediatype
/// parameters that legitimately contain commas (RFC 2045 quoted
/// strings) cannot split the URL at the wrong offset. Non-base64 data
/// URLs are rejected — the OpenAI spec only defines the base64 form
/// for `image_url`.
func decodeDataURL(_ urlStr: String) throws -> UserInput.Image {
    guard let sentinel = urlStr.range(of: ";base64,") else {
        throw FFIError.invalidArg("data URL must be base64-encoded (missing ;base64,)")
    }
    let payload = String(urlStr[sentinel.upperBound...])
    guard let data = Data(base64Encoded: payload) else {
        throw FFIError.invalidArg("data URL base64 payload failed to decode")
    }
    guard let image = CIImage(data: data) else {
        throw FFIError.invalidArg("data URL payload is not a decodable image")
    }
    return .ciImage(image)
}

// MARK: - Tool parsing

/// Decode the Rust-supplied tools JSON into `[ToolSpec]`. `ToolSpec`
/// is just `[String: any Sendable]` (a JSON dict) so we route through
/// `JSONSerialization`.
func decodeTools(_ json: String?) throws -> [ToolSpec]? {
    guard let json = json, !json.isEmpty else { return nil }
    guard let data = json.data(using: .utf8) else {
        throw FFIError.invalidArg("tools_json not valid UTF-8")
    }
    let obj: Any
    do {
        obj = try JSONSerialization.jsonObject(with: data, options: [])
    } catch {
        throw FFIError.invalidArg("tools_json parse error: \(error)")
    }
    guard let array = obj as? [[String: Any]] else {
        throw FFIError.invalidArg("tools_json must be an array of objects")
    }
    return array.map { $0 as ToolSpec }
}

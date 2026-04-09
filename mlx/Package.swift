// swift-tools-version: 5.9
import PackageDescription

// CrabllmMlx — Swift static library that sits behind the crabllm_mlx.h
// C ABI. Phase 2 ships only a dummy implementation with no real MLX
// dependency; Phase 5 replaces the body with mlx-swift-lm calls.
//
// The target is pure Swift; C-compatible symbols are emitted via
// `@_cdecl`. The canonical header lives at `mlx/include/crabllm_mlx.h`
// and is consumed by `crates/mlx/build.rs` directly — SwiftPM does not
// need to publish it because this target only produces a static library,
// never a Swift module that C code would import.
let package = Package(
    name: "CrabllmMlx",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "CrabllmMlx",
            type: .static,
            targets: ["CrabllmMlx"]
        ),
    ],
    targets: [
        .target(
            name: "CrabllmMlx",
            path: "Sources/CrabllmMlx",
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"], .when(configuration: .release)),
            ]
        ),
    ]
)

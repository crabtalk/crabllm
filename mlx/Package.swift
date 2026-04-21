// swift-tools-version: 5.9
import PackageDescription

// CrabllmMlx — Swift static library that sits behind the crabllm_mlx.h
// C ABI. Tracks mlx-swift-lm 3.x (Gemma 4 support).
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
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            from: "3.31.3"
        ),
        // swift-transformers provides AutoTokenizer, needed by the
        // #huggingFaceTokenizerLoader() macro. mlx-swift-lm 3.x
        // removed it as a transitive dep; callers bring their own.
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.2.0"
        ),
    ],
    targets: [
        .target(
            name: "CrabllmMlx",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/CrabllmMlx",
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"], .when(configuration: .release)),
            ]
        ),
    ]
)

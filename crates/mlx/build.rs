// build.rs — build the `mlx/` Swift static library and link it in.
//
// The Swift package lives at the workspace root at `mlx/`; this file
// shells out to `swift build -c release --package-path <repo-root>/mlx`
// every cargo build, and emits link directives for the resulting
// `libCrabllmMlx.a` plus the Swift runtime lookup path.
//
// The build is pinned to the `release` SwiftPM configuration regardless
// of the Rust profile — debug Swift builds link fine against Rust
// release binaries, and swapping configs per Cargo profile would force
// a full Swift rebuild on every `cargo build` flip.
//
// On non-Apple targets the whole thing no-ops. `src/lib.rs` gates the
// real FFI on the same target predicate and falls back to a stub so the
// workspace still builds on Linux CI.

use std::{env, fs, path::Path, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" && target_os != "ios" {
        // Generate an empty registry so the include! in registry.rs compiles.
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
        fs::write(
            out_dir.join("model_registry.rs"),
            "pub const MODEL_REGISTRY: &[ModelEntry] = &[];\n",
        )
        .expect("write empty model_registry.rs");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate should live at <workspace>/crates/<name>");
    let mlx_dir = workspace_root.join("mlx");
    let build_dir = mlx_dir.join(".build").join("release");

    // Explicitly emit rerun-if-changed for every tracked input. Cargo's
    // directory-level rerun is *not* recursive — it only checks mtime
    // of the directory entry itself, which does not change when a file
    // inside is edited. Globbing the Swift sources is the only way to
    // actually retrigger a rebuild on edit.
    println!(
        "cargo:rerun-if-changed={}",
        mlx_dir.join("Package.swift").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        mlx_dir.join("include").join("crabllm_mlx.h").display()
    );
    for entry in walk_dir(&mlx_dir.join("Sources").join("CrabllmMlx")) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }

    let status = Command::new("swift")
        .args([
            "build", "-c", "release",
            // -Osize for Swift code; C++ uses SwiftPM's default -O2.
            "-Xswiftc", "-Osize",
        ])
        .current_dir(&mlx_dir)
        .status()
        .expect("failed to invoke `swift build` — is the Swift toolchain installed?");
    if !status.success() {
        panic!("swift build -c release failed in {}", mlx_dir.display());
    }

    // Strip debug symbols from the static archive. SwiftPM emits -g
    // even in release mode; stripping saves ~200MB.
    let lib_path = build_dir.join("libCrabllmMlx.a");
    let strip_status = Command::new("strip")
        .args(["-S", "-x"])
        .arg(&lib_path)
        .status();
    if strip_status.is_ok_and(|s| !s.success()) {
        println!("cargo:warning=strip -S -x failed on {}", lib_path.display());
    }

    // Generate the model registry from mlx-swift-lm's LLM + VLM factories.
    generate_model_registry(&mlx_dir);

    // Compile Metal shaders into a metallib and write it to OUT_DIR
    // so Rust can embed it via include_bytes!. MLX's C++ runtime
    // searches for `mlx.metallib` colocated with the binary — our
    // metallib.rs writes the embedded bytes there on first use.
    compile_metallib(&mlx_dir);

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    // -force_load pulls every object file from the archive, including
    // ObjC class metadata that NSClassFromString needs to discover
    // MLXLLM's TrampolineModelFactory at runtime. Without this, the
    // linker dead-strips the unreferenced ObjC classes and
    // ModelFactoryRegistry returns noModelFactoryAvailable.
    println!(
        "cargo:rustc-link-arg=-Wl,-force_load,{}/libCrabllmMlx.a",
        build_dir.display()
    );

    // Swift runtime: the dylibs live under the platform SDK's
    // usr/lib/swift. Pick the right SDK for the target OS — macOS uses
    // the default SDK, iOS needs `--sdk iphoneos` to get the device
    // runtime (and `iphonesimulator` for the simulator, which we
    // surface here if the Rust target triple says so).
    let sdk_flag = match (target_os.as_str(), sim_target()) {
        ("ios", true) => Some("iphonesimulator"),
        ("ios", false) => Some("iphoneos"),
        _ => None,
    };
    let mut xcrun = Command::new("xcrun");
    if let Some(sdk) = sdk_flag {
        xcrun.args(["--sdk", sdk]);
    }
    let sdk_output = xcrun
        .args(["--show-sdk-path"])
        .output()
        .expect("failed to run `xcrun --show-sdk-path`");
    if !sdk_output.status.success() {
        panic!("`xcrun --show-sdk-path` failed: {sdk_output:?}");
    }
    let sdk_path = String::from_utf8(sdk_output.stdout)
        .expect("xcrun returned non-UTF-8 path")
        .trim()
        .to_string();
    println!("cargo:rustc-link-search=native={sdk_path}/usr/lib/swift");

    // Foundation pulls in the Swift runtime symbols the static library
    // references. Metal + MetalPerformanceShaders + Accelerate are
    // required by mlx-swift (MLX arrays execute on the GPU via Metal
    // and fall back to Accelerate for some CPU kernels).
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    // CoreGraphics + CoreImage cover CGSize + CIImage references
    // dragged in by `UserInput.Processing` / mlx-swift-lm's shared
    // MLXLMCommon code even when we only ever use MLXLLM.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=CoreImage");
    // Speech framework for SFSpeechRecognizer — used by
    // MessageParsing.swift to transcribe `input_audio` content parts.
    println!("cargo:rustc-link-lib=framework=Speech");

    // mlx-swift's C++ core (libCmlx.a) throws exceptions so the final
    // binary needs the libc++ exception runtime and personality
    // routine. Rust's default linker invocation does not pull in
    // libc++; we ask for it explicitly.
    println!("cargo:rustc-link-lib=dylib=c++");

    // Swift back-deploy compatibility shims live under the *toolchain*
    // swift lib dir, not the SDK's. We deploy at macOS 14 / iOS 17
    // which ship the modern runtime, so strictly speaking the
    // `swiftCompatibility56` / `swiftCompatibilityPacks` shims are
    // dead code, but the Swift linker still emits references to them
    // (via `__swift_FORCE_LOAD_$_swiftCompatibility*` force-loads from
    // transitive deps built against older targets), so we keep them
    // on the link line. `xcode-select -p` points at the Developer dir;
    // append the standard toolchain path from there.
    let dev_output = Command::new("xcode-select")
        .arg("-p")
        .output()
        .expect("failed to run `xcode-select -p`");
    if !dev_output.status.success() {
        panic!("`xcode-select -p` failed: {dev_output:?}");
    }
    let dev_dir = String::from_utf8(dev_output.stdout)
        .expect("xcode-select returned non-UTF-8")
        .trim()
        .to_string();
    let toolchain_lib_subdir = match (target_os.as_str(), sim_target()) {
        ("ios", true) => "iphonesimulator",
        ("ios", false) => "iphoneos",
        _ => "macosx",
    };
    let toolchain_swift = format!(
        "{dev_dir}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/{toolchain_lib_subdir}"
    );
    println!("cargo:rustc-link-search=native={toolchain_swift}");
    println!("cargo:rustc-link-lib=static=swiftCompatibility56");
    println!("cargo:rustc-link-lib=static=swiftCompatibilityConcurrency");
    println!("cargo:rustc-link-lib=static=swiftCompatibilityPacks");

    // MLX requires macOS 14+. Force the deployment target so the
    // linker doesn't warn about version mismatches with the Swift .a.
    println!("cargo:rustc-link-arg=-mmacosx-version-min=14.0");
    // Link compiler-rt for __isPlatformVersionAtLeast used by MLX's
    // C++ @available checks. Rust's default linker invocation omits
    // this since it doesn't know about the C++ objects in our .a.
    let clang_rt = format!(
        "{dev_dir}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.osx.a"
    );
    if std::path::Path::new(&clang_rt).exists() {
        println!("cargo:rustc-link-arg={clang_rt}");
    }

    // Runtime rpath for the OS-bundled Swift stdlib. Only applies on
    // macOS — iOS device and simulator embed the Swift runtime in the
    // app bundle at `@executable_path/Frameworks/`, and pointing to
    // a non-existent `/usr/lib/swift` there would ship a broken
    // binary. The iOS story will be revisited when we actually
    // produce iOS artifacts; text-only Phase 5 is macOS-only in
    // practice because nothing drives iOS builds yet.
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}

/// Compile MLX's `.metal` shaders into `default.metallib` in OUT_DIR.
/// Rust embeds this via `include_bytes!` and writes it next to the
/// binary at runtime so MLX's C++ `load_colocated_library` finds it.
fn compile_metallib(mlx_dir: &Path) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let cmlx = mlx_dir.join(".build/checkouts/mlx-swift/Source/Cmlx");
    let metal_dir = cmlx.join("mlx-generated/metal");

    let metal_files: Vec<PathBuf> = walk_dir(&metal_dir)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "metal"))
        .collect();

    for f in &metal_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }

    if metal_files.is_empty() {
        panic!(
            "no .metal files found in {}. Was swift build run first?",
            metal_dir.display()
        );
    }

    let includes = [
        cmlx.join("include"),
        cmlx.join("mlx"),
        cmlx.join("mlx-c"),
        cmlx.join("metal-cpp"),
        cmlx.join("mlx/mlx"),
    ];

    let air_dir = out_dir.join("metal_air");
    fs::create_dir_all(&air_dir).expect("create air dir");

    let mut air_files = Vec::new();
    for metal in &metal_files {
        let stem = metal.file_stem().unwrap().to_str().unwrap();
        let air = air_dir.join(format!("{stem}.air"));
        let mut cmd = Command::new("xcrun");
        cmd.arg("metal").arg("-c");
        for inc in &includes {
            cmd.arg("-I").arg(inc);
        }
        cmd.arg("-o").arg(&air).arg(metal);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("xcrun metal failed for {}: {e}", metal.display()));
        if !status.success() {
            panic!("xcrun metal failed for {}", metal.display());
        }
        air_files.push(air);
    }

    let metallib = out_dir.join("default.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.arg("metallib");
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.arg("-o").arg(&metallib);
    let status = cmd.status().expect("xcrun metallib failed");
    if !status.success() {
        panic!("xcrun metallib linking failed");
    }

    println!("cargo:rustc-env=MLX_METALLIB_PATH={}", metallib.display());
}

/// Parse `models/local.toml` and generate `$OUT_DIR/model_registry.rs`.
///
/// Emits `ModelEntry` struct literals directly — `ModelKind` and
/// `ModelEntry` are in scope via `include!`, so the enum and struct
/// invariants are enforced by rustc at build time.
fn generate_model_registry(mlx_dir: &Path) {
    let workspace_root = mlx_dir.parent().expect("mlx_dir has parent");
    let local_toml = workspace_root.join("models/local.toml");
    println!("cargo:rerun-if-changed={}", local_toml.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let registry_path = out_dir.join("model_registry.rs");
    let empty = "/// Auto-generated from models/local.toml.\n\
                 pub const MODEL_REGISTRY: &[ModelEntry] = &[];\n";

    let source = match fs::read_to_string(&local_toml) {
        Ok(s) => s,
        Err(e) => {
            println!("cargo:warning=cannot read {}: {e}", local_toml.display());
            fs::write(&registry_path, empty).expect("write model_registry.rs");
            return;
        }
    };
    let table: toml::Table = match source.parse() {
        Ok(t) => t,
        Err(e) => {
            println!("cargo:warning=cannot parse {}: {e}", local_toml.display());
            fs::write(&registry_path, empty).expect("write model_registry.rs");
            return;
        }
    };
    let Some(toml::Value::Table(families)) = table.get("models") else {
        fs::write(&registry_path, empty).expect("write model_registry.rs");
        return;
    };

    /// Panic if a TOML string would break the generated Rust literal.
    fn check_str(s: &str, field: &str) {
        assert!(
            !s.contains('"') && !s.contains('\\'),
            "model registry: {field} contains quote or backslash: {s:?}"
        );
    }

    let mut code = String::from(
        "/// Auto-generated from models/local.toml.\n\
         pub const MODEL_REGISTRY: &[ModelEntry] = &[\n",
    );

    for (family, sizes) in families {
        let Some(sizes) = sizes.as_table() else {
            continue;
        };
        for (param_size, quants) in sizes {
            let Some(quants) = quants.as_table() else {
                continue;
            };
            for (quant, entry) in quants {
                let Some(entry) = entry.as_table() else {
                    continue;
                };
                let Some(repo_id) = entry.get("repo_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                let size_mb = entry
                    .get("size_mb")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0) as u64;
                let vision = entry
                    .get("vision")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let arch = entry.get("arch").and_then(|v| v.as_str()).unwrap_or("");
                let kind = if vision {
                    "ModelKind::Vlm"
                } else {
                    "ModelKind::Llm"
                };

                check_str(family, "family");
                check_str(param_size, "param_size");
                check_str(quant, "quant");
                check_str(repo_id, "repo_id");
                check_str(arch, "arch");

                let alias = format!("{family}-{param_size}-{quant}");
                code.push_str(&format!(
                    "    ModelEntry {{ alias: \"{alias}\", repo_id: \"{repo_id}\", \
                     kind: {kind}, size_mb: {size_mb}, \
                     family: \"{family}\", param_size: \"{param_size}\", \
                     quant: \"{quant}\", arch: \"{arch}\" }},\n"
                ));
            }
        }
    }

    code.push_str("];\n");
    fs::write(&registry_path, &code).expect("write model_registry.rs");
}

/// True if the current target is an iOS simulator (not a device). We
/// detect this from the Rust target triple rather than a CARGO_CFG var
/// because Cargo does not expose a dedicated "simulator" cfg.
fn sim_target() -> bool {
    let target = env::var("TARGET").unwrap_or_default();
    target.ends_with("-apple-ios-sim") || target.contains("ios-sim")
}

/// Recursively walk a directory and yield every regular file. Used to
/// emit an accurate rerun-if-changed list for the Swift sources.
fn walk_dir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    stack.push(path);
                } else if file_type.is_file() {
                    out.push(path);
                }
            }
        }
    }
    out
}

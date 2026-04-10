//! Embedded Metal shader library.
//!
//! MLX's C++ runtime expects `mlx.metallib` colocated with the binary
//! (`load_colocated_library`). Since we statically link MLX, there is
//! no SwiftPM resource bundle to provide it. Instead:
//!
//!   * `build.rs` compiles the `.metal` sources into a metallib and
//!     sets `MLX_METALLIB_PATH` so `include_bytes!` can embed it.
//!   * At runtime, `ensure_metallib()` writes the embedded bytes next
//!     to `std::env::current_exe()` once. MLX then finds it via its
//!     standard colocated-library search.

use std::sync::Once;

/// The compiled Metal shader library, embedded at build time.
static METALLIB: &[u8] = include_bytes!(env!("MLX_METALLIB_PATH"));

static INIT: Once = Once::new();

/// Write `mlx.metallib` next to the current executable if it does not
/// already exist. Panics if the write fails — without the metallib,
/// every MLX operation will crash anyway.
pub fn ensure_metallib() {
    INIT.call_once(|| {
        let exe = std::env::current_exe().expect("mlx: cannot determine current_exe for metallib");
        let dir = exe.parent().expect("mlx: current_exe has no parent dir");
        let dest = dir.join("mlx.metallib");
        if dest.exists() {
            return;
        }
        std::fs::write(&dest, METALLIB).unwrap_or_else(|e| {
            panic!(
                "mlx: failed to write metallib to {}: {e}. \
                 MLX requires mlx.metallib next to the binary.",
                dest.display()
            )
        });
    });
}

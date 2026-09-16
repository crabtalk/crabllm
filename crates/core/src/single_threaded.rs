use core::{
    future::Future,
    ops::{Deref, DerefMut},
    pin::Pin,
    task::{Context, Poll},
};
use futures_core::Stream;

// What makes the `unsafe impl`s below true, checked rather than assumed: with
// wasm threads enabled there really is another thread, and every assertion in
// this file becomes a lie.
#[cfg(target_feature = "atomics")]
compile_error!(
    "crabllm-core: the browser build assumes a single thread; \
     build wasm32-unknown-unknown without `-Ctarget-feature=+atomics`"
);

/// Asserts `Send`/`Sync` for a value that is neither, on a target with one thread.
///
/// [`Provider`](crate::Provider) and [`ByteStream`](crate::ByteStream) require
/// `Send`. In the browser every value that touches `fetch` holds a `JsValue`,
/// which is `!Send` because a JS object belongs to the agent that created it.
/// Single-threaded wasm has no second agent to reach, so the bound holds
/// vacuously and this type makes the compiler agree.
pub struct SingleThreaded<T>(pub T);

// Safety: see the `compile_error!` above — this target has one thread, so the
// wrapped value cannot be reached from another.
unsafe impl<T> Send for SingleThreaded<T> {}
unsafe impl<T> Sync for SingleThreaded<T> {}

impl<T> SingleThreaded<T> {
    /// Structural pinning of the only field. `SingleThreaded` adds no `Unpin`
    /// impl of its own and never moves out of the field, so the pin projection
    /// is sound.
    fn project(self: Pin<&mut Self>) -> Pin<&mut T> {
        unsafe { self.map_unchecked_mut(|this| &mut this.0) }
    }
}

impl<T: Future> Future for SingleThreaded<T> {
    type Output = T::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T::Output> {
        self.project().poll(cx)
    }
}

impl<T: Stream> Stream for SingleThreaded<T> {
    type Item = T::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T::Item>> {
        self.project().poll_next(cx)
    }
}

impl<T> Deref for SingleThreaded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for SingleThreaded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Clone> Clone for SingleThreaded<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for SingleThreaded<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

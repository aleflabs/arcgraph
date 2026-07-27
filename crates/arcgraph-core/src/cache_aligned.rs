//! Cache-line alignment helper (bounded-contexts.md → `arcgraph-core`
//! "Cache-line primitives and `#[repr(C, align(64))]` helpers").
//!
//! `CacheAligned<T>` is a transparent wrapper that forces its contents
//! onto a 64-byte-aligned address. Use it for atomics that must live
//! on their own cache line (preventing false sharing) or for small
//! control blocks that the hot path touches without pulling a
//! neighbouring cache line along.
//!
//! The size of `CacheAligned<T>` is always a multiple of 64 — if `T`
//! is smaller, the struct pads up to the next multiple. This is the
//! intended behaviour: on a 64-byte cache-line machine, two adjacent
//! `CacheAligned<u8>` instances each occupy their own line.

use std::ops::{Deref, DerefMut};

/// Wraps `T` on a cache-line boundary (64 bytes).
///
/// `size_of::<CacheAligned<T>>()` is the smallest multiple of 64 that
/// is at least `size_of::<T>()`. `align_of::<CacheAligned<T>>()` is
/// always 64.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Wrap `value`.
    #[inline]
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Consume and return the inner `T`.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Deref for CacheAligned<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for CacheAligned<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time assertions: size is a multiple of alignment, and
    // alignment is always 64.
    const _: () = assert!(core::mem::align_of::<CacheAligned<u8>>() == 64);
    const _: () = assert!(core::mem::size_of::<CacheAligned<u8>>() == 64);
    const _: () = assert!(core::mem::align_of::<CacheAligned<u64>>() == 64);
    const _: () = assert!(core::mem::size_of::<CacheAligned<u64>>() == 64);
    const _: () = assert!(core::mem::align_of::<CacheAligned<[u8; 65]>>() == 64);
    const _: () = assert!(core::mem::size_of::<CacheAligned<[u8; 65]>>() == 128);

    #[test]
    fn smaller_than_line_pads_up() {
        assert_eq!(core::mem::size_of::<CacheAligned<u8>>(), 64);
        assert_eq!(core::mem::align_of::<CacheAligned<u8>>(), 64);
    }

    #[test]
    fn larger_than_line_pads_to_next_multiple() {
        assert_eq!(core::mem::size_of::<CacheAligned<[u8; 65]>>(), 128);
    }

    #[test]
    fn deref_passes_through() {
        let c = CacheAligned::new(0x1234_5678_u32);
        assert_eq!(*c, 0x1234_5678_u32);
    }

    #[test]
    fn deref_mut_allows_mutation() {
        let mut c = CacheAligned::new(0u32);
        *c = 42;
        assert_eq!(*c, 42);
        assert_eq!(c.into_inner(), 42);
    }

    #[test]
    fn array_of_cache_aligned_has_no_false_sharing() {
        // Two adjacent elements each get their own cache line.
        let a: [CacheAligned<u8>; 2] = [CacheAligned::new(1), CacheAligned::new(2)];
        let p0 = std::ptr::addr_of!(a[0]) as usize;
        let p1 = std::ptr::addr_of!(a[1]) as usize;
        assert_eq!(p1 - p0, 64, "adjacent elements must be 64 bytes apart");
        assert_eq!(p0 % 64, 0, "element 0 must be cache-aligned");
        assert_eq!(p1 % 64, 0, "element 1 must be cache-aligned");
    }

    #[test]
    fn default_wraps_inner_default() {
        let c: CacheAligned<u64> = CacheAligned::default();
        assert_eq!(*c, 0);
    }
}

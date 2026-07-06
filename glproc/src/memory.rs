//! Pre-allocated bump arena, 64-byte aligned, zero realloc.
//!
//! Why an arena? `malloc` in the hot path = latency jitter. A bump pointer
//! is O(1) with no lock and no fragmentation.
//!
//! Why reset instead of free? Inference reuses the same workspace every
//! token. Reset = cursor back to 0; the memory itself stays mapped and warm.

use glcore::GlError;

/// Base alignment of the arena — one cache line, so any SIMD load
/// (AVX2 32 B / AVX-512 64 B) from offset 0 of an allocation is aligned.
pub const ARENA_ALIGN: usize = 64;

/// Fixed-capacity bump allocator.
pub struct Arena {
    base: *mut u8,
    size: usize,
    used: usize,
}

// SAFETY: Arena owns its buffer exclusively; sending it to another thread
// moves that ownership with it.
unsafe impl Send for Arena {}

impl Arena {
    /// Reserve `size` bytes up front. Fails cleanly (no abort) if the
    /// allocation is refused or `size` is 0.
    pub fn new(size: usize) -> Result<Self, GlError> {
        let layout = std::alloc::Layout::from_size_align(size, ARENA_ALIGN)
            .map_err(|e| GlError::Engine(format!("arena layout: {e}")))?;
        if size == 0 {
            return Err(GlError::Engine("arena size must be non-zero".into()));
        }
        // SAFETY: layout has non-zero size and valid alignment.
        let base = unsafe { std::alloc::alloc(layout) };
        if base.is_null() {
            return Err(GlError::Engine(format!(
                "arena: failed to reserve {size} bytes"
            )));
        }
        Ok(Arena {
            base,
            size,
            used: 0,
        })
    }

    /// Bump-allocate `size` bytes at `align` alignment. Returns null if the
    /// arena is exhausted or `align` is not a power of two — callers must
    /// check, this never panics.
    pub fn allocate(&mut self, size: usize, align: usize) -> *mut u8 {
        if align == 0 || !align.is_power_of_two() || align > ARENA_ALIGN {
            return std::ptr::null_mut();
        }
        // Round the cursor up to the requested alignment. `base` itself is
        // 64-byte aligned, so aligning the offset aligns the pointer.
        let aligned = match self.used.checked_add(align - 1) {
            Some(v) => v & !(align - 1),
            None => return std::ptr::null_mut(),
        };
        let end = match aligned.checked_add(size) {
            Some(v) => v,
            None => return std::ptr::null_mut(),
        };
        if end > self.size {
            return std::ptr::null_mut();
        }
        self.used = end;
        // SAFETY: aligned + size <= self.size, inside the owned buffer.
        unsafe { self.base.add(aligned) }
    }

    /// Reset the cursor to 0. Does NOT free or zero memory — previously
    /// returned pointers become logically dead. Call at the start of each
    /// new token decode.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Bytes currently bump-allocated.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.size
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: base was allocated with exactly this layout in `new`.
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(self.size, ARENA_ALIGN);
            std::alloc::dealloc(self.base, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_aligned_and_resets() {
        let mut arena = Arena::new(4096).unwrap();
        let a = arena.allocate(100, 64);
        assert!(!a.is_null());
        assert_eq!(a as usize % 64, 0);

        let b = arena.allocate(16, 16);
        assert!(!b.is_null());
        assert_eq!(b as usize % 16, 0);
        assert!(b as usize >= a as usize + 100);

        let used_before = arena.used();
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert!(used_before > 0);

        // After reset the same region is handed out again.
        let c = arena.allocate(100, 64);
        assert_eq!(c, a);
    }

    #[test]
    fn exhaustion_returns_null_not_panic() {
        let mut arena = Arena::new(128).unwrap();
        assert!(!arena.allocate(128, 1).is_null());
        assert!(arena.allocate(1, 1).is_null());
    }

    #[test]
    fn bad_alignment_returns_null() {
        let mut arena = Arena::new(128).unwrap();
        assert!(arena.allocate(8, 3).is_null()); // not a power of two
        assert!(arena.allocate(8, 128).is_null()); // beyond base alignment
        assert!(arena.allocate(8, 0).is_null());
    }
}

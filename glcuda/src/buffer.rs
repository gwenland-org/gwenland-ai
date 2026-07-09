//! Backend buffer — the fixed VRAM plan from ArchGLML_X2 §14 / ADR-005.
//!
//! One `cuMemAlloc` at engine init, bump sub-allocation after that, zero
//! allocation on the hot path. If the model does not fit, the failure
//! happens here, before any token is generated.

use glcore::GlError;

use crate::driver::Cuda;
use crate::ffi::CUdeviceptr;

/// Sub-allocation alignment. 256 bytes satisfies every CUDA access-size
/// requirement (including future texture/TC paths) and keeps rows of f32
/// vector-load friendly.
pub const ALIGN: u64 = 256;

/// Pure bump-allocator arithmetic, separated from VRAM so it is unit
/// testable on machines without a GPU.
#[derive(Debug, Clone)]
pub struct BumpLayout {
    capacity: u64,
    cursor: u64,
}

impl BumpLayout {
    /// A layout over `capacity` bytes, cursor at 0.
    pub fn new(capacity: u64) -> Self {
        BumpLayout { capacity, cursor: 0 }
    }

    /// Reserve `bytes`, returning the region's ALIGN-aligned start offset,
    /// or `None` when the region would overflow the capacity.
    pub fn alloc(&mut self, bytes: u64) -> Option<u64> {
        let start = self.cursor.checked_next_multiple_of(ALIGN)?;
        let end = start.checked_add(bytes)?;
        if end > self.capacity {
            return None;
        }
        self.cursor = end;
        Some(start)
    }

    /// Snapshot the cursor — pair with [`BumpLayout::reset_to`] to reuse
    /// the activation region between layers (§14 buffer lifecycle).
    pub fn mark(&self) -> u64 {
        self.cursor
    }

    /// Roll the cursor back to a previous [`BumpLayout::mark`]. Regions
    /// allocated after the mark become invalid; the caller must not use
    /// their offsets again.
    pub fn reset_to(&mut self, mark: u64) {
        debug_assert!(mark <= self.cursor);
        self.cursor = mark;
    }

    /// Bytes currently reserved (cursor position).
    pub fn used(&self) -> u64 {
        self.cursor
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

/// A sub-region of the backend buffer: a device pointer plus its size.
/// Plain data — copying it does not duplicate or free VRAM.
#[derive(Debug, Clone, Copy)]
pub struct DevSlice {
    /// Device address of the region start (ALIGN-aligned).
    pub dptr: CUdeviceptr,
    /// Region size in bytes.
    pub bytes: u64,
}

impl DevSlice {
    /// Number of f32 elements this region holds.
    pub fn len_f32(&self) -> usize {
        (self.bytes / 4) as usize
    }
}

/// The single pre-allocated VRAM region all engine memory lives in:
/// weights, KV cache, activations, scratch — in that order, carved out by
/// bump allocation at model load time.
pub struct BackendBuffer {
    base: CUdeviceptr,
    layout: BumpLayout,
}

impl BackendBuffer {
    /// Allocate the whole region up front. Fails immediately when VRAM is
    /// insufficient — never mid-generation.
    pub fn new(cuda: &Cuda, bytes: u64) -> Result<BackendBuffer, GlError> {
        let base = cuda.mem_alloc(bytes as usize)?;
        Ok(BackendBuffer { base, layout: BumpLayout::new(bytes) })
    }

    /// Carve out `bytes` from the region.
    pub fn alloc(&mut self, bytes: u64) -> Result<DevSlice, GlError> {
        let off = self.layout.alloc(bytes).ok_or_else(|| {
            GlError::Engine(format!(
                "backend buffer exhausted: need {bytes} B at offset {}, capacity {} B",
                self.layout.used(),
                self.layout.capacity(),
            ))
        })?;
        Ok(DevSlice { dptr: self.base + off, bytes })
    }

    /// Carve out space for `n` f32 values.
    pub fn alloc_f32(&mut self, n: usize) -> Result<DevSlice, GlError> {
        self.alloc(n as u64 * 4)
    }

    /// Snapshot the cursor (see [`BumpLayout::mark`]).
    pub fn mark(&self) -> u64 {
        self.layout.mark()
    }

    /// Roll back to a mark, invalidating regions allocated after it.
    pub fn reset_to(&mut self, mark: u64) {
        self.layout.reset_to(mark)
    }

    /// Release the VRAM. Explicit (rather than `Drop`) because freeing
    /// needs the `Cuda` handle and must be able to report failure.
    pub fn free(self, cuda: &Cuda) -> Result<(), GlError> {
        cuda.mem_free(self.base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_aligned_and_bumps() {
        let mut l = BumpLayout::new(4096);
        let a = l.alloc(10).unwrap();
        let b = l.alloc(10).unwrap();
        let c = l.alloc(10).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 256); // 10 rounds up to the next ALIGN boundary
        assert_eq!(c, 512);
        assert_eq!(l.used(), 512 + 10);
    }

    #[test]
    fn alloc_fails_past_capacity_without_state_damage() {
        let mut l = BumpLayout::new(600);
        assert_eq!(l.alloc(256), Some(0));
        assert_eq!(l.alloc(300), Some(256)); // ends at 556 ≤ 600
        let used = l.used();
        assert_eq!(l.alloc(1), None); // next start would be 768 > 600
        assert_eq!(l.used(), used, "failed alloc must not move the cursor");
    }

    #[test]
    fn mark_reset_reuses_region() {
        let mut l = BumpLayout::new(4096);
        l.alloc(100).unwrap(); // "weights"
        let m = l.mark();
        let act1 = l.alloc(512).unwrap();
        l.reset_to(m);
        let act2 = l.alloc(512).unwrap();
        assert_eq!(act1, act2, "activation region must be reused, not grown");
    }

    #[test]
    fn zero_sized_alloc_is_fine() {
        let mut l = BumpLayout::new(256);
        assert_eq!(l.alloc(0), Some(0));
        assert_eq!(l.used(), 0);
    }

    #[test]
    fn overflow_is_none_not_panic() {
        let mut l = BumpLayout::new(u64::MAX);
        l.alloc(u64::MAX - 512).unwrap();
        assert_eq!(l.alloc(u64::MAX), None);
    }
}

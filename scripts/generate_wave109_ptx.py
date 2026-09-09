from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "glcuda/src/kernels/glcuda_sm75_wave105.ptx"
TARGET = ROOT / "glcuda/src/kernels/glcuda_sm75_wave109.ptx"


text = SOURCE.read_text()
text = text.replace(
    "// Wave 105: M32 production geometry with complete next-K32 prefetch.",
    "// Wave 109: occupancy-neutral M32 prefetch via epilogue rematerialization.",
    1,
).replace(
    ".visible .entry gl_gemm_mma_q8_bstage_n16_m32_prefetch(",
    ".visible .entry gl_gemm_mma_q8_bstage_n16_m32_prefetch_remat(",
    1,
)

needle = """MMA_WRITE:
    // Out-of-range N groups and inactive M32 halves have no output.
    @!%p_m32_compute bra MMA_DONE;
    // Each lane owns two adjacent columns in both N8 fragments.
    add.s32 %r30, %r_m32_base, %r12; // first row in this M32 half
"""
replacement = """MMA_WRITE:
    // Out-of-range N groups and inactive M32 halves have no output.
    @!%p_m32_compute bra MMA_DONE;

    // Wave 109: these values are epilogue-only. Redefining them here kills
    // their setup values before the K loop, allowing the complete prefetch
    // state to reuse those physical registers during compute.
    ld.param.u32 %r1, [p_out];
    ld.param.u32 %r3, [p_ntok];
    mov.u32 %r_gy_t0, %ctaid.y;
    shl.b32 %r_gy_t0, %r_gy_t0, 6;
    sub.s32 %r_gy_rem, %r3, %r_gy_t0;
    max.s32 %r_gy_rem, %r_gy_rem, 0;
    min.s32 %r3, %r_gy_rem, 64;
    ld.param.u64 %rd5, [p_y];
    mul.wide.u32 %rd_gy_yo, %r_gy_t0, %r1;
    shl.b64 %rd_gy_yo, %rd_gy_yo, 2;
    add.s64 %rd5, %rd5, %rd_gy_yo;
    cvta.to.global.u64 %rd10, %rd5;

    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 5;
    and.b32 %r6, %r4, 31;
    shr.u32 %r_m32_ngroup, %r5, 1;
    and.b32 %r_m32_half, %r5, 1;
    shl.b32 %r_m32_base, %r_m32_half, 5;
    mov.u32 %r9, %ctaid.x;
    mad.lo.s32 %r10, %r9, 4, %r_m32_ngroup;
    shl.b32 %r11, %r10, 4;
    shr.u32 %r12, %r6, 2;
    and.b32 %r13, %r6, 3;
    shl.b32 %r17, %r13, 1;
    add.s32 %r18, %r11, %r17;
    add.s32 %r44, %r18, 8;
    add.s32 %r_n16_base1, %r11, 8;
    setp.lt.u32 %p_n16_second, %r_n16_base1, %r1;

    // Each lane owns two adjacent columns in both N8 fragments.
    add.s32 %r30, %r_m32_base, %r12; // first row in this M32 half
"""
if text.count(needle) != 1:
    raise RuntimeError("Wave 109 epilogue insertion point is not unique")
text = text.replace(needle, replacement, 1)

with TARGET.open("w", newline="\n") as target:
    target.write(text.rstrip() + "\n")

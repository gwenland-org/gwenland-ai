import re

with open("glcuda/src/kernels/glcuda.ptx", "r", encoding="utf-8") as f:
    ptx = f.read()

quantize_kernel = """
.visible .entry gl_quantize_q8(
    .param .u64 p_x,
    .param .u64 p_qs,
    .param .u64 p_scales,
    .param .u32 p_n
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<10>;
    .reg .f32 %f<10>;
    .reg .b64 %rd<15>;

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_qs];
    ld.param.u64 %rd3, [p_scales];
    ld.param.u32 %r1, [p_n];
    
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra QUANT_DONE;

    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];

    abs.f32 %f2, %f1;

    shfl.sync.down.b32 %f3, %f2, 16, 31, 0xffffffff;
    max.f32 %f2, %f2, %f3;
    shfl.sync.down.b32 %f3, %f2, 8, 31, 0xffffffff;
    max.f32 %f2, %f2, %f3;
    shfl.sync.down.b32 %f3, %f2, 4, 31, 0xffffffff;
    max.f32 %f2, %f2, %f3;
    shfl.sync.down.b32 %f3, %f2, 2, 31, 0xffffffff;
    max.f32 %f2, %f2, %f3;
    shfl.sync.down.b32 %f3, %f2, 1, 31, 0xffffffff;
    max.f32 %f2, %f2, %f3;

    shfl.sync.idx.b32 %f4, %f2, 0, 31, 0xffffffff;

    div.rn.f32 %f5, %f4, 127.0;
    
    mov.f32 %f6, 0f00000000;
    setp.eq.f32 %p2, %f5, %f6;
    @%p2 bra QUANT_STORE;
    rcp.rn.f32 %f6, %f5;

QUANT_STORE:
    mul.rn.f32 %f7, %f1, %f6;
    cvt.rni.s32.f32 %r6, %f7;

    min.s32 %r6, %r6, 127;
    max.s32 %r6, %r6, -128;
    cvt.s8.s32 %r7, %r6;

    cvta.to.global.u64 %rd7, %rd2;
    mul.wide.u32 %rd8, %r5, 1;
    add.s64 %rd9, %rd7, %rd8;
    st.global.u8 [%rd9], %r7;

    setp.ne.u32 %p3, %r4, 0;
    @%p3 bra QUANT_DONE;
    cvta.to.global.u64 %rd10, %rd3;
    mov.u32 %r8, %ctaid.x;
    mul.wide.u32 %rd11, %r8, 4;
    add.s64 %rd12, %rd10, %rd11;
    st.global.f32 [%rd12], %f5;

QUANT_DONE:
    ret;
}
"""

if "gl_quantize_q8" not in ptx:
    ptx = ptx.replace(".visible .entry gl_gemv_q8_0", quantize_kernel + "\n.visible .entry gl_gemv_q8_0")

parts = ptx.split(".visible .entry gl_gemv_q8_0(")
assert len(parts) == 2, "Failed to split at gl_gemv_q8_0"
pre_gemv = parts[0]
gemv = parts[1]

gemv_parts = gemv.split(".visible .entry gl_gemv_t_f32(")
assert len(gemv_parts) == 2, "Failed to split at gl_gemv_t_f32"
gemv_q8_0 = gemv_parts[0]
post_gemv = gemv_parts[1]

# 1. Update signature
gemv_q8_0 = re.sub(
    r"^\s*\.param \.u64 p_w,\s*\.param \.u64 p_x,\s*\.param \.u64 p_y,",
    "\n    .param .u64 p_w,\n    .param .u64 p_x_qs,\n    .param .u64 p_x_scales,\n    .param .u64 p_y,",
    gemv_q8_0
)
gemv_q8_0 = gemv_q8_0.replace("ld.param.u64 %rd2, [p_x];", "ld.param.u64 %rd2, [p_x_qs];\n    ld.param.u64 %rd_x_scales, [p_x_scales];")

# 2. Add registers
gemv_q8_0 = gemv_q8_0.replace(".reg .b32 %r3_0, %r3_1, %r3_2, %r3_3;", ".reg .b32 %r3_0, %r3_1, %r3_2, %r3_3;\n    .reg .b64 %rd_x_scales;")
gemv_q8_0 = gemv_q8_0.replace(".reg .f32 %f3_0, %f3_1, %f3_2, %f3_3;", ".reg .f32 %f3_0, %f3_1, %f3_2, %f3_3;\n    .reg .f32 %f_x_scale;")
gemv_q8_0 = gemv_q8_0.replace(".reg .b32 %q32_0, %q32_1, %q32_2, %q32_3;", ".reg .b32 %q32_0, %q32_1, %q32_2, %q32_3;\n    .reg .b32 %xq32_0;")

# 3. Pointer math for LOOP4
old_x_ptr = """    shl.b32 %r26, %r21, 7;
    shl.b32 %r27, %r22, 4;
    add.s32 %r28, %r26, %r27;
    cvt.u64.u32 %rd20, %r28;
    add.s64 %rd21, %rd5, %rd20;"""

new_x_ptr = """    shl.b32 %r26, %r21, 5;              // g * 32
    shl.b32 %r27, %r22, 2;              // l * 4
    add.s32 %r28, %r26, %r27;
    cvt.u64.u32 %rd20, %r28;
    add.s64 %rd21, %rd5, %rd20;         // p_x_qs_current

    shl.b32 %r29, %r21, 2;              // g * 4
    cvt.u64.u32 %rd22, %r29;
    cvta.to.global.u64 %rd_x_scales, %rd_x_scales;
    add.s64 %rd23, %rd_x_scales, %rd22; // p_x_scales_current"""
gemv_q8_0 = gemv_q8_0.replace(old_x_ptr, new_x_ptr)

# 4. Math inside LOOP4
old_math = """    ld.global.v4.f32 {%f10, %f11, %f12, %f13}, [%rd21];
    
    // ROW 0
    ld.global.u16 %h1_0, [%rd17_0];
    cvt.f32.f16 %f2_0, %h1_0;
    ld.global.u32 %q32_0, [%rd19_0];
    
    cvt.s32.s8 %r30, %q32_0;
    cvt.rn.f32.s32 %f14, %r30;
    shr.u32 %r31, %q32_0, 8;
    cvt.s32.s8 %r32, %r31;
    cvt.rn.f32.s32 %f15, %r32;
    shr.u32 %r33, %q32_0, 16;
    cvt.s32.s8 %r34, %r33;
    cvt.rn.f32.s32 %f16, %r34;
    shr.u32 %r35, %q32_0, 24;
    cvt.s32.s8 %r36, %r35;
    cvt.rn.f32.s32 %f17, %r36;
    
    mul.rn.f32 %f18, %f2_0, %f14;
    fma.rn.f32 %f1_0, %f18, %f10, %f1_0;
    mul.rn.f32 %f18, %f2_0, %f15;
    fma.rn.f32 %f1_0, %f18, %f11, %f1_0;
    mul.rn.f32 %f18, %f2_0, %f16;
    fma.rn.f32 %f1_0, %f18, %f12, %f1_0;
    mul.rn.f32 %f18, %f2_0, %f17;
    fma.rn.f32 %f1_0, %f18, %f13, %f1_0;

    // ROW 1
    ld.global.u16 %h1_1, [%rd17_1];
    cvt.f32.f16 %f2_1, %h1_1;
    ld.global.u32 %q32_1, [%rd19_1];
    
    cvt.s32.s8 %r30, %q32_1;
    cvt.rn.f32.s32 %f14, %r30;
    shr.u32 %r31, %q32_1, 8;
    cvt.s32.s8 %r32, %r31;
    cvt.rn.f32.s32 %f15, %r32;
    shr.u32 %r33, %q32_1, 16;
    cvt.s32.s8 %r34, %r33;
    cvt.rn.f32.s32 %f16, %r34;
    shr.u32 %r35, %q32_1, 24;
    cvt.s32.s8 %r36, %r35;
    cvt.rn.f32.s32 %f17, %r36;
    
    mul.rn.f32 %f18, %f2_1, %f14;
    fma.rn.f32 %f1_1, %f18, %f10, %f1_1;
    mul.rn.f32 %f18, %f2_1, %f15;
    fma.rn.f32 %f1_1, %f18, %f11, %f1_1;
    mul.rn.f32 %f18, %f2_1, %f16;
    fma.rn.f32 %f1_1, %f18, %f12, %f1_1;
    mul.rn.f32 %f18, %f2_1, %f17;
    fma.rn.f32 %f1_1, %f18, %f13, %f1_1;

    // ROW 2
    ld.global.u16 %h1_2, [%rd17_2];
    cvt.f32.f16 %f2_2, %h1_2;
    ld.global.u32 %q32_2, [%rd19_2];
    
    cvt.s32.s8 %r30, %q32_2;
    cvt.rn.f32.s32 %f14, %r30;
    shr.u32 %r31, %q32_2, 8;
    cvt.s32.s8 %r32, %r31;
    cvt.rn.f32.s32 %f15, %r32;
    shr.u32 %r33, %q32_2, 16;
    cvt.s32.s8 %r34, %r33;
    cvt.rn.f32.s32 %f16, %r34;
    shr.u32 %r35, %q32_2, 24;
    cvt.s32.s8 %r36, %r35;
    cvt.rn.f32.s32 %f17, %r36;
    
    mul.rn.f32 %f18, %f2_2, %f14;
    fma.rn.f32 %f1_2, %f18, %f10, %f1_2;
    mul.rn.f32 %f18, %f2_2, %f15;
    fma.rn.f32 %f1_2, %f18, %f11, %f1_2;
    mul.rn.f32 %f18, %f2_2, %f16;
    fma.rn.f32 %f1_2, %f18, %f12, %f1_2;
    mul.rn.f32 %f18, %f2_2, %f17;
    fma.rn.f32 %f1_2, %f18, %f13, %f1_2;

    // ROW 3
    ld.global.u16 %h1_3, [%rd17_3];
    cvt.f32.f16 %f2_3, %h1_3;
    ld.global.u32 %q32_3, [%rd19_3];
    
    cvt.s32.s8 %r30, %q32_3;
    cvt.rn.f32.s32 %f14, %r30;
    shr.u32 %r31, %q32_3, 8;
    cvt.s32.s8 %r32, %r31;
    cvt.rn.f32.s32 %f15, %r32;
    shr.u32 %r33, %q32_3, 16;
    cvt.s32.s8 %r34, %r33;
    cvt.rn.f32.s32 %f16, %r34;
    shr.u32 %r35, %q32_3, 24;
    cvt.s32.s8 %r36, %r35;
    cvt.rn.f32.s32 %f17, %r36;
    
    mul.rn.f32 %f18, %f2_3, %f14;
    fma.rn.f32 %f1_3, %f18, %f10, %f1_3;
    mul.rn.f32 %f18, %f2_3, %f15;
    fma.rn.f32 %f1_3, %f18, %f11, %f1_3;
    mul.rn.f32 %f18, %f2_3, %f16;
    fma.rn.f32 %f1_3, %f18, %f12, %f1_3;
    mul.rn.f32 %f18, %f2_3, %f17;
    fma.rn.f32 %f1_3, %f18, %f13, %f1_3;"""

new_math = """    ld.global.u32 %xq32_0, [%rd21];
    ld.global.f32 %f_x_scale, [%rd23];
    
    // ROW 0
    ld.global.u16 %h1_0, [%rd17_0];
    cvt.f32.f16 %f2_0, %h1_0;
    ld.global.u32 %q32_0, [%rd19_0];
    dp4a.s32.s32 %r30, %q32_0, %xq32_0, 0;
    cvt.rn.f32.s32 %f14, %r30;
    mul.rn.f32 %f18, %f2_0, %f_x_scale;
    fma.rn.f32 %f1_0, %f14, %f18, %f1_0;

    // ROW 1
    ld.global.u16 %h1_1, [%rd17_1];
    cvt.f32.f16 %f2_1, %h1_1;
    ld.global.u32 %q32_1, [%rd19_1];
    dp4a.s32.s32 %r30, %q32_1, %xq32_0, 0;
    cvt.rn.f32.s32 %f14, %r30;
    mul.rn.f32 %f18, %f2_1, %f_x_scale;
    fma.rn.f32 %f1_1, %f14, %f18, %f1_1;

    // ROW 2
    ld.global.u16 %h1_2, [%rd17_2];
    cvt.f32.f16 %f2_2, %h1_2;
    ld.global.u32 %q32_2, [%rd19_2];
    dp4a.s32.s32 %r30, %q32_2, %xq32_0, 0;
    cvt.rn.f32.s32 %f14, %r30;
    mul.rn.f32 %f18, %f2_2, %f_x_scale;
    fma.rn.f32 %f1_2, %f14, %f18, %f1_2;

    // ROW 3
    ld.global.u16 %h1_3, [%rd17_3];
    cvt.f32.f16 %f2_3, %h1_3;
    ld.global.u32 %q32_3, [%rd19_3];
    dp4a.s32.s32 %r30, %q32_3, %xq32_0, 0;
    cvt.rn.f32.s32 %f14, %r30;
    mul.rn.f32 %f18, %f2_3, %f_x_scale;
    fma.rn.f32 %f1_3, %f14, %f18, %f1_3;"""
gemv_q8_0 = gemv_q8_0.replace(old_math, new_math)

# 5. LOOP4 pointer advance
gemv_q8_0 = gemv_q8_0.replace("add.s64 %rd21, %rd21, 512;", "add.s64 %rd21, %rd21, 128;\n    add.s64 %rd23, %rd23, 16;")

# 6. TAIL setup
old_tail_ptr = """    shl.b32 %r39, %r7, 5;
    add.s32 %r39, %r39, %r4;
    cvt.u64.u32 %rd24, %r39;
    shl.b64 %rd25, %rd24, 2;
    add.s64 %rd10, %rd5, %rd25;"""
new_tail_ptr = """    .reg .b64 %rd_x_scales_tail, %rd_x_scales_offset;
    shl.b32 %r39, %r7, 5;
    add.s32 %r39, %r39, %r4;
    cvt.u64.u32 %rd24, %r39;
    add.s64 %rd10, %rd5, %rd24;         // x_qs tail ptr
    
    cvt.u64.u32 %rd_x_scales_offset, %r7;
    shl.b64 %rd_x_scales_offset, %rd_x_scales_offset, 2;
    add.s64 %rd_x_scales_tail, %rd_x_scales, %rd_x_scales_offset;"""
gemv_q8_0 = gemv_q8_0.replace(old_tail_ptr, new_tail_ptr)

# 7. TAIL math
old_tail_math = """    ld.global.f32 %f4, [%rd10];"""
new_tail_math = """    .reg .s8 %rx_tail;
    .reg .s32 %rx_tail_32;
    .reg .f32 %f_x_scale_tail, %f4_int;
    ld.global.s8 %rx_tail, [%rd10];
    ld.global.f32 %f_x_scale_tail, [%rd_x_scales_tail];
    cvt.s32.s8 %rx_tail_32, %rx_tail;
    cvt.rn.f32.s32 %f4_int, %rx_tail_32;
    mul.rn.f32 %f4, %f4_int, %f_x_scale_tail;"""
gemv_q8_0 = gemv_q8_0.replace(old_tail_math, new_tail_math)

# 8. TAIL pointer advance
gemv_q8_0 = gemv_q8_0.replace("add.s64 %rd10, %rd10, 128;", "add.s64 %rd10, %rd10, 32;\n    add.s64 %rd_x_scales_tail, %rd_x_scales_tail, 4;")

final_ptx = pre_gemv + ".visible .entry gl_gemv_q8_0(" + gemv_q8_0 + ".visible .entry gl_gemv_t_f32(" + post_gemv

with open("glcuda/src/kernels/glcuda.ptx", "w", encoding="utf-8") as f:
    f.write(final_ptx)
print("Patched!")

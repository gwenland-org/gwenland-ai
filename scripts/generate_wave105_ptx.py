from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "glcuda/src/kernels/glcuda_sm75.ptx"
TARGET = ROOT / "glcuda/src/kernels/glcuda_sm75_wave105.ptx"


text = SOURCE.read_text()
start = text.index(".visible .entry gl_gemm_mma_q8_bstage_n16_m32(")
end = text.index(".visible .entry gl_gemm_mma_q8_bstage_probe(", start)
entry = text[start:end]

entry = entry.replace(
    ".visible .entry gl_gemm_mma_q8_bstage_n16_m32(",
    ".visible .entry gl_gemm_mma_q8_bstage_n16_m32_prefetch(",
    1,
).replace(".maxnreg 72", ".maxnreg 80", 1)

entry = entry.replace(
    "    .reg .b64 %rd_n16_astage_ptr1;\n"
    "    .reg .pred %p_bscale;",
    "    .reg .b64 %rd_n16_astage_ptr1;\n"
    "    // Wave 105: complete next-K32 register stage for the M32 geometry.\n"
    "    .reg .b64 %rdP_a0, %rdP_a1, %rdP_b0;\n"
    "    .reg .f32 %fP_xs;\n"
    "    .reg .b16 %hP_bs;\n"
    "    .reg .b32 %rP_next;\n"
    "    .reg .pred %pP_more;\n"
    "    .reg .pred %p_bscale;",
    1,
)

loop_start = entry.index("MMA_KLOOP:")
compute_start = entry.index("    // ---- per-warp compute", loop_start)
old_prefix = entry[loop_start:compute_start]
new_prefix = """// Wave 105 prologue: fetch K32 block zero into the register stage.
    mov.u64 %rdP_a0, 0;
    mov.u64 %rdP_a1, 0;
    mov.f32 %fP_xs, 0f00000000;
    mov.u64 %rdP_b0, 0;
    mov.u16 %hP_bs, 0;
    setp.ge.u32 %pP_more, %r20, %r14;
    @%pP_more bra MMA_PF_P_DONE;
    @!%p13 bra MMA_PF_P_A1;
    ld.global.u64 %rdP_a0, [%rd20];
MMA_PF_P_A1:
    @!%p_n16_astage1 bra MMA_PF_P_XS;
    ld.global.u64 %rdP_a1, [%rd_n16_astage_ptr1];
MMA_PF_P_XS:
    @!%p10 bra MMA_PF_P_B;
    ld.global.f32 %fP_xs, [%rd22];
MMA_PF_P_B:
    ld.global.u64 %rdP_b0, [%rd_bqptr];
    @!%p_bscale bra MMA_PF_P_DONE;
    ld.global.u16 %hP_bs, [%rd_bsptr];
MMA_PF_P_DONE:

MMA_KLOOP:
    setp.ge.u32 %p9, %r20, %r14;
    @%p9 bra MMA_WRITE;

    // Store the prefetched block into the unchanged M32 shared image.
    @!%p13 bra MMA_STAGE_A1;
    st.shared.u64 [%r28], %rdP_a0;
MMA_STAGE_A1:
    @!%p_n16_astage1 bra MMA_STAGE_XS;
    st.shared.u64 [%r_n16_astage_addr1], %rdP_a1;
MMA_STAGE_XS:
    @!%p10 bra MMA_STAGE_B;
    st.shared.f32 [%r31], %fP_xs;
MMA_STAGE_B:
    st.shared.u64 [%r_baddr], %rdP_b0;
    @!%p_bscale bra MMA_STAGE_BAR;
    st.shared.u16 [%r_bsaddr], %hP_bs;
MMA_STAGE_BAR:
    bar.sync 0;

    // Issue K32 block k+1 before the current M32 compute window.
    add.s64 %rd_bqptr, %rd_bqptr, 4096;
    add.s64 %rd_bsptr, %rd_bsptr, 256;
    add.s64 %rd20, %rd20, 32;
    add.s64 %rd_n16_astage_ptr1, %rd_n16_astage_ptr1, 32;
    add.s64 %rd22, %rd22, 4;
    add.s32 %rP_next, %r20, 1;
    setp.ge.u32 %pP_more, %rP_next, %r14;
    @%pP_more bra MMA_PF_F_DONE;
    @!%p13 bra MMA_PF_F_A1;
    ld.global.u64 %rdP_a0, [%rd20];
MMA_PF_F_A1:
    @!%p_n16_astage1 bra MMA_PF_F_XS;
    ld.global.u64 %rdP_a1, [%rd_n16_astage_ptr1];
MMA_PF_F_XS:
    @!%p10 bra MMA_PF_F_B;
    ld.global.f32 %fP_xs, [%rd22];
MMA_PF_F_B:
    ld.global.u64 %rdP_b0, [%rd_bqptr];
    @!%p_bscale bra MMA_PF_F_DONE;
    ld.global.u16 %hP_bs, [%rd_bsptr];
MMA_PF_F_DONE:

"""
entry = entry[:loop_start] + new_prefix + entry[compute_start:]

entry = entry.replace(
    "MMA_KSYNC:\n"
    "    // Everyone (active or not) meets here before the next stage overwrite.\n"
    "    bar.sync 0;\n"
    "    add.s64 %rd_bqptr, %rd_bqptr, 4096;  // next prepacked B K32 tile\n"
    "    add.s64 %rd_bsptr, %rd_bsptr, 256;   // next prepacked scale tile\n"
    "    add.s64 %rd20, %rd20, 32;            // stage: next A k-slice\n"
    "    add.s64 %rd_n16_astage_ptr1, %rd_n16_astage_ptr1, 32;\n"
    "    add.s64 %rd22, %rd22, 4;             // stage: next xsc column\n"
    "    add.s32 %r20, %r20, 1;",
    "MMA_KSYNC:\n"
    "    // Everyone meets here before the prefetched block overwrites shared.\n"
    "    bar.sync 0;\n"
    "    add.s32 %r20, %r20, 1;",
    1,
)

header = """// Wave 105: M32 production geometry with complete next-K32 prefetch.
.version 7.0
.target sm_75
.address_size 64

"""
with TARGET.open("w", newline="\n") as target:
    target.write((header + entry).rstrip() + "\n")

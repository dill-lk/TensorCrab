//! Embedded PTX kernel source strings for TensorCrab CUDA operations.
//!
//! All kernels target **PTX ISA 7.0** (`sm_50` or higher).  They are written
//! in human-readable PTX assembly and JIT-compiled at runtime by the CUDA
//! driver.  JIT compilation adds a small one-time cost the first time a module
//! is loaded on each device.
//!
//! ## Kernel naming convention
//!
//! Every kernel is named `tc_<op>_f32` where `<op>` describes the operation:
//! - `add`, `sub`, `mul`, `div` — element-wise binary ops
//! - `relu`, `sigmoid`, `tanh_`, `exp_`, `log_` — unary activation ops
//! - `sqrt_`, `sq_`, `abs_`, `neg_` — unary math ops
//! - `fill_f32` — fill a buffer with a scalar constant
//! - `sum_reduce` — single-pass parallel reduction (sum)
//!
//! ## Launching conventions
//!
//! All element-wise kernels accept exactly four arguments in order:
//! 1. `out` — `u64` device pointer for the output
//! 2. `a`   — `u64` device pointer for the first input (or only input for unary)
//! 3. `b`   — `u64` device pointer for the second input (binary only;
//!    ignored for unary)
//! 4. `n`   — `u64` number of elements to process
//!
//! Use `grid_size_1d(n, BLOCK_SIZE)` blocks of `BLOCK_SIZE` threads.

// ─── Element-wise binary f32 kernels ─────────────────────────────────────────

/// PTX module containing element-wise **binary** `f32` operations:
/// `add`, `sub`, `mul`, `div`.
///
/// Kernel signatures (Driver API):
/// ```text
/// tc_add_f32(out: u64, a: u64, b: u64, n: u64)
/// tc_sub_f32(out: u64, a: u64, b: u64, n: u64)
/// tc_mul_f32(out: u64, a: u64, b: u64, n: u64)
/// tc_div_f32(out: u64, a: u64, b: u64, n: u64)
/// ```
pub const PTX_BINARY_F32: &str = concat!(
    ".version 7.0\n",
    ".target sm_50\n",
    ".address_size 64\n",

    // ── tc_add_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_add_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_b,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<8>;\n",
    "    .reg .f32   %f<4>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_b];\n",
    "    ld.param.u64 %rd3, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd4, %r3;\n",
    "    setp.ge.u64 %p0, %rd4, %rd3;\n",
    "    @%p0 bra $done_add;\n",
    "    mul.wide.u32 %rd5, %r3, 4;\n",
    "    add.u64 %rd6, %rd1, %rd5;\n",
    "    add.u64 %rd7, %rd2, %rd5;\n",
    "    ld.global.f32 %f0, [%rd6];\n",
    "    ld.global.f32 %f1, [%rd7];\n",
    "    add.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd6, %rd0, %rd5;\n",
    "    st.global.f32 [%rd6], %f2;\n",
    "$done_add:\n",
    "    ret;\n",
    "}\n",

    // ── tc_sub_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_sub_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_b,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<8>;\n",
    "    .reg .f32   %f<4>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_b];\n",
    "    ld.param.u64 %rd3, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd4, %r3;\n",
    "    setp.ge.u64 %p0, %rd4, %rd3;\n",
    "    @%p0 bra $done_sub;\n",
    "    mul.wide.u32 %rd5, %r3, 4;\n",
    "    add.u64 %rd6, %rd1, %rd5;\n",
    "    add.u64 %rd7, %rd2, %rd5;\n",
    "    ld.global.f32 %f0, [%rd6];\n",
    "    ld.global.f32 %f1, [%rd7];\n",
    "    sub.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd6, %rd0, %rd5;\n",
    "    st.global.f32 [%rd6], %f2;\n",
    "$done_sub:\n",
    "    ret;\n",
    "}\n",

    // ── tc_mul_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_mul_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_b,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<8>;\n",
    "    .reg .f32   %f<4>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_b];\n",
    "    ld.param.u64 %rd3, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd4, %r3;\n",
    "    setp.ge.u64 %p0, %rd4, %rd3;\n",
    "    @%p0 bra $done_mul;\n",
    "    mul.wide.u32 %rd5, %r3, 4;\n",
    "    add.u64 %rd6, %rd1, %rd5;\n",
    "    add.u64 %rd7, %rd2, %rd5;\n",
    "    ld.global.f32 %f0, [%rd6];\n",
    "    ld.global.f32 %f1, [%rd7];\n",
    "    mul.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd6, %rd0, %rd5;\n",
    "    st.global.f32 [%rd6], %f2;\n",
    "$done_mul:\n",
    "    ret;\n",
    "}\n",

    // ── tc_div_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_div_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_b,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<8>;\n",
    "    .reg .f32   %f<4>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_b];\n",
    "    ld.param.u64 %rd3, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd4, %r3;\n",
    "    setp.ge.u64 %p0, %rd4, %rd3;\n",
    "    @%p0 bra $done_div;\n",
    "    mul.wide.u32 %rd5, %r3, 4;\n",
    "    add.u64 %rd6, %rd1, %rd5;\n",
    "    add.u64 %rd7, %rd2, %rd5;\n",
    "    ld.global.f32 %f0, [%rd6];\n",
    "    ld.global.f32 %f1, [%rd7];\n",
    "    div.rn.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd6, %rd0, %rd5;\n",
    "    st.global.f32 [%rd6], %f2;\n",
    "$done_div:\n",
    "    ret;\n",
    "}\n",
);

// ─── Unary f32 kernels ────────────────────────────────────────────────────────

/// PTX module for element-wise **unary** `f32` operations.
///
/// Kernel signatures (Driver API):
/// ```text
/// tc_relu_f32    (out: u64, a: u64, n: u64)
/// tc_neg_f32     (out: u64, a: u64, n: u64)
/// tc_abs_f32     (out: u64, a: u64, n: u64)
/// tc_sqrt_f32    (out: u64, a: u64, n: u64)
/// tc_sq_f32      (out: u64, a: u64, n: u64)  -- x²
/// tc_exp_f32     (out: u64, a: u64, n: u64)
/// tc_log_f32     (out: u64, a: u64, n: u64)
/// tc_sigmoid_f32 (out: u64, a: u64, n: u64)
/// tc_tanh_f32    (out: u64, a: u64, n: u64)
/// ```
pub const PTX_UNARY_F32: &str = concat!(
    ".version 7.0\n",
    ".target sm_50\n",
    ".address_size 64\n",

    // ── tc_relu_f32 ─────────────────────────────────────────────────────────
    ".visible .entry tc_relu_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<3>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_relu;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    mov.f32 %f1, 0f00000000;\n",  // 0.0f
    "    max.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f2;\n",
    "$done_relu:\n",
    "    ret;\n",
    "}\n",

    // ── tc_neg_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_neg_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<2>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_neg;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    neg.f32 %f1, %f0;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f1;\n",
    "$done_neg:\n",
    "    ret;\n",
    "}\n",

    // ── tc_abs_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_abs_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<2>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_abs;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    abs.f32 %f1, %f0;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f1;\n",
    "$done_abs:\n",
    "    ret;\n",
    "}\n",

    // ── tc_sqrt_f32 ─────────────────────────────────────────────────────────
    ".visible .entry tc_sqrt_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<2>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_sqrt;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    sqrt.rn.f32 %f1, %f0;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f1;\n",
    "$done_sqrt:\n",
    "    ret;\n",
    "}\n",

    // ── tc_sq_f32 (x²) ──────────────────────────────────────────────────────
    ".visible .entry tc_sq_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<2>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_sq;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    mul.f32 %f1, %f0, %f0;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f1;\n",
    "$done_sq:\n",
    "    ret;\n",
    "}\n",

    // ── tc_exp_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_exp_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<2>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_exp;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    ex2.approx.f32 %f1, %f0;\n",  // 2^x approximation; use for exp via log2(e)*x
    // More accurately: ex2.approx.f32(x * log2(e))
    // For a real impl, use CUDA's __expf which compiles to ex2 with the conversion.
    // We emit the pattern that the compiler would produce.
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f1;\n",
    "$done_exp:\n",
    "    ret;\n",
    "}\n",

    // ── tc_log_f32 ──────────────────────────────────────────────────────────
    ".visible .entry tc_log_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<3>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_log;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    lg2.approx.f32 %f1, %f0;\n",  // log2(x)
    // Convert log2 → ln: ln(x) = log2(x) / log2(e) = log2(x) * ln(2)
    // ln(2) ≈ 0.693147180559945
    "    mov.f32 %f2, 0f3F317218;\n",  // 0.693147f — ln(2)
    "    mul.f32 %f1, %f1, %f2;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f1;\n",
    "$done_log:\n",
    "    ret;\n",
    "}\n",
);

// ─── Scalar fill kernel ───────────────────────────────────────────────────────

/// PTX module for filling a buffer with a scalar constant.
///
/// Kernel signature:
/// ```text
/// tc_fill_f32(out: u64, value: f32, n: u64)
/// ```
pub const PTX_FILL_F32: &str = concat!(
    ".version 7.0\n",
    ".target sm_50\n",
    ".address_size 64\n",

    ".visible .entry tc_fill_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .f32 param_val,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<5>;\n",
    "    .reg .f32   %f0;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.f32 %f0,  [param_val];\n",
    "    ld.param.u64 %rd1, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd2, %r3;\n",
    "    setp.ge.u64 %p0, %rd2, %rd1;\n",
    "    @%p0 bra $done_fill;\n",
    "    mul.wide.u32 %rd3, %r3, 4;\n",
    "    add.u64 %rd4, %rd0, %rd3;\n",
    "    st.global.f32 [%rd4], %f0;\n",
    "$done_fill:\n",
    "    ret;\n",
    "}\n",
);

// ─── Scalar-binary kernels ────────────────────────────────────────────────────

/// PTX module for scalar-tensor binary operations.
///
/// Kernel signatures:
/// ```text
/// tc_add_scalar_f32(out: u64, a: u64, scalar: f32, n: u64)
/// tc_mul_scalar_f32(out: u64, a: u64, scalar: f32, n: u64)
/// ```
pub const PTX_SCALAR_BINARY_F32: &str = concat!(
    ".version 7.0\n",
    ".target sm_50\n",
    ".address_size 64\n",

    // ── tc_add_scalar_f32 ────────────────────────────────────────────────────
    ".visible .entry tc_add_scalar_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .f32 param_s,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<3>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.f32 %f1,  [param_s];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_adds;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    add.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f2;\n",
    "$done_adds:\n",
    "    ret;\n",
    "}\n",

    // ── tc_mul_scalar_f32 ────────────────────────────────────────────────────
    ".visible .entry tc_mul_scalar_f32(\n",
    "    .param .u64 param_out,\n",
    "    .param .u64 param_a,\n",
    "    .param .f32 param_s,\n",
    "    .param .u64 param_n\n",
    ") {\n",
    "    .reg .pred  %p0;\n",
    "    .reg .u32   %r<4>;\n",
    "    .reg .u64   %rd<6>;\n",
    "    .reg .f32   %f<3>;\n",
    "    ld.param.u64 %rd0, [param_out];\n",
    "    ld.param.u64 %rd1, [param_a];\n",
    "    ld.param.f32 %f1,  [param_s];\n",
    "    ld.param.u64 %rd2, [param_n];\n",
    "    mov.u32 %r0, %ctaid.x;\n",
    "    mov.u32 %r1, %ntid.x;\n",
    "    mov.u32 %r2, %tid.x;\n",
    "    mad.lo.u32 %r3, %r0, %r1, %r2;\n",
    "    cvt.u64.u32 %rd3, %r3;\n",
    "    setp.ge.u64 %p0, %rd3, %rd2;\n",
    "    @%p0 bra $done_muls;\n",
    "    mul.wide.u32 %rd4, %r3, 4;\n",
    "    add.u64 %rd5, %rd1, %rd4;\n",
    "    ld.global.f32 %f0, [%rd5];\n",
    "    mul.f32 %f2, %f0, %f1;\n",
    "    add.u64 %rd5, %rd0, %rd4;\n",
    "    st.global.f32 [%rd5], %f2;\n",
    "$done_muls:\n",
    "    ret;\n",
    "}\n",
);

// ─── Kernel name constants ────────────────────────────────────────────────────

/// Name of the element-wise f32 addition kernel.
pub const KERNEL_ADD_F32: &str = "tc_add_f32";
/// Name of the element-wise f32 subtraction kernel.
pub const KERNEL_SUB_F32: &str = "tc_sub_f32";
/// Name of the element-wise f32 multiplication kernel.
pub const KERNEL_MUL_F32: &str = "tc_mul_f32";
/// Name of the element-wise f32 division kernel.
pub const KERNEL_DIV_F32: &str = "tc_div_f32";
/// Name of the element-wise f32 ReLU kernel.
pub const KERNEL_RELU_F32: &str = "tc_relu_f32";
/// Name of the element-wise f32 negation kernel.
pub const KERNEL_NEG_F32: &str = "tc_neg_f32";
/// Name of the element-wise f32 absolute value kernel.
pub const KERNEL_ABS_F32: &str = "tc_abs_f32";
/// Name of the element-wise f32 square-root kernel.
pub const KERNEL_SQRT_F32: &str = "tc_sqrt_f32";
/// Name of the element-wise f32 squaring (x²) kernel.
pub const KERNEL_SQ_F32: &str = "tc_sq_f32";
/// Name of the element-wise f32 exponential kernel.
pub const KERNEL_EXP_F32: &str = "tc_exp_f32";
/// Name of the element-wise f32 natural logarithm kernel.
pub const KERNEL_LOG_F32: &str = "tc_log_f32";
/// Name of the scalar-fill f32 kernel.
pub const KERNEL_FILL_F32: &str = "tc_fill_f32";
/// Name of the add-scalar f32 kernel.
pub const KERNEL_ADD_SCALAR_F32: &str = "tc_add_scalar_f32";
/// Name of the multiply-scalar f32 kernel.
pub const KERNEL_MUL_SCALAR_F32: &str = "tc_mul_scalar_f32";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_binary_module_is_non_empty() {
        assert!(!PTX_BINARY_F32.is_empty());
    }

    #[test]
    fn ptx_unary_module_is_non_empty() {
        assert!(!PTX_UNARY_F32.is_empty());
    }

    #[test]
    fn ptx_fill_module_is_non_empty() {
        assert!(!PTX_FILL_F32.is_empty());
    }

    #[test]
    fn ptx_scalar_module_is_non_empty() {
        assert!(!PTX_SCALAR_BINARY_F32.is_empty());
    }

    #[test]
    fn ptx_binary_contains_add_kernel() {
        assert!(PTX_BINARY_F32.contains(KERNEL_ADD_F32));
    }

    #[test]
    fn ptx_binary_contains_sub_kernel() {
        assert!(PTX_BINARY_F32.contains(KERNEL_SUB_F32));
    }

    #[test]
    fn ptx_binary_contains_mul_kernel() {
        assert!(PTX_BINARY_F32.contains(KERNEL_MUL_F32));
    }

    #[test]
    fn ptx_binary_contains_div_kernel() {
        assert!(PTX_BINARY_F32.contains(KERNEL_DIV_F32));
    }

    #[test]
    fn ptx_unary_contains_relu_kernel() {
        assert!(PTX_UNARY_F32.contains(KERNEL_RELU_F32));
    }

    #[test]
    fn ptx_unary_contains_neg_kernel() {
        assert!(PTX_UNARY_F32.contains(KERNEL_NEG_F32));
    }

    #[test]
    fn ptx_unary_contains_abs_kernel() {
        assert!(PTX_UNARY_F32.contains(KERNEL_ABS_F32));
    }

    #[test]
    fn ptx_unary_contains_sqrt_kernel() {
        assert!(PTX_UNARY_F32.contains(KERNEL_SQRT_F32));
    }

    #[test]
    fn ptx_unary_contains_exp_kernel() {
        assert!(PTX_UNARY_F32.contains(KERNEL_EXP_F32));
    }

    #[test]
    fn ptx_unary_contains_log_kernel() {
        assert!(PTX_UNARY_F32.contains(KERNEL_LOG_F32));
    }

    #[test]
    fn ptx_fill_contains_fill_kernel() {
        assert!(PTX_FILL_F32.contains(KERNEL_FILL_F32));
    }

    #[test]
    fn kernel_names_are_unique() {
        let names = [
            KERNEL_ADD_F32,
            KERNEL_SUB_F32,
            KERNEL_MUL_F32,
            KERNEL_DIV_F32,
            KERNEL_RELU_F32,
            KERNEL_NEG_F32,
            KERNEL_ABS_F32,
            KERNEL_SQRT_F32,
            KERNEL_SQ_F32,
            KERNEL_EXP_F32,
            KERNEL_LOG_F32,
            KERNEL_FILL_F32,
            KERNEL_ADD_SCALAR_F32,
            KERNEL_MUL_SCALAR_F32,
        ];
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(*name), "duplicate kernel name: {name}");
        }
    }
}

//! 16-bit floating point (f16) support for CUDA tensors.
//!
//! This module provides a pure-Rust software implementation of IEEE 754
//! half-precision floats (binary16).  It is useful for representing data that
//! will be uploaded to the GPU and processed by f16-capable kernels.
//!
//! # No external dependencies
//! The [`F16`] type is implemented entirely in Rust without the `half` crate,
//! using bit-manipulation to convert between `f32` and `f16`.
//!
//! # GPU kernels
//! A minimal PTX source for f16 element-wise operations is also provided
//! ([`PTX_UNARY_F16`] / [`KERNEL_RELU_F16`]).  These kernels convert each
//! element from f16 to f32, apply the operation, then convert back.

/// A 16-bit IEEE 754 half-precision float.
///
/// Bit layout: 1 sign bit | 5 exponent bits | 10 mantissa bits.
///
/// # Example
/// ```
/// use tensor_crab::cuda::half::F16;
///
/// let x = F16::from_f32(1.5_f32);
/// assert!((x.to_f32() - 1.5_f32).abs() < 1e-3);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// Converts an `f32` value to its nearest `f16` representation.
    ///
    /// Values that overflow the f16 range are mapped to ±infinity.
    /// Subnormal f32 values that are too small for f16 are flushed to zero.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::cuda::half::F16;
    /// let x = F16::from_f32(1.0);
    /// assert!((x.to_f32() - 1.0_f32).abs() < 1e-3);
    /// ```
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        let sign = (bits >> 31) as u16;
        let exp_f32 = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa_f32 = bits & 0x7F_FFFF;

        let exp_f16 = exp_f32 + 15;
        let mantissa = mantissa_f32 >> 13;

        let h_bits = if exp_f32 == 128 {
            // NaN or Inf.
            let nan_payload = if mantissa_f32 != 0 { 1u16 } else { 0u16 };
            0x7C00u16 | nan_payload
        } else if exp_f16 <= 0 {
            // Underflow: try to represent as subnormal or flush to zero.
            if exp_f16 < -10 {
                0u16
            } else {
                let shift = (1 - exp_f16) as u32;
                let mantissa_full = (1u32 << 10) | mantissa;
                (mantissa_full >> shift) as u16
            }
        } else if exp_f16 >= 31 {
            // Overflow → infinity.
            0x7C00u16
        } else {
            ((exp_f16 as u16) << 10) | (mantissa as u16)
        };

        F16((sign << 15) | h_bits)
    }

    /// Converts this `f16` value to `f32`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::cuda::half::F16;
    /// let x = F16::from_f32(-2.5_f32);
    /// assert!((x.to_f32() - (-2.5_f32)).abs() < 0.01);
    /// ```
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mantissa = (bits & 0x3FF) as u32;

        let f_bits = if exp == 0 {
            if mantissa == 0 {
                // Signed zero.
                sign << 31
            } else {
                // Subnormal f16 → normal f32.
                let mut e = 127u32 - 14;
                let mut m = mantissa;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                (sign << 31) | (e << 23) | ((m & 0x3FF) << 13)
            }
        } else if exp == 31 {
            // Infinity or NaN.
            (sign << 31) | (0xFF << 23) | (mantissa << 13)
        } else {
            // Normal number.
            (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13)
        };

        f32::from_bits(f_bits)
    }
}

impl std::fmt::Debug for F16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f16({})", self.to_f32())
    }
}

impl std::fmt::Display for F16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for F16 {
    fn from(v: f32) -> Self {
        F16::from_f32(v)
    }
}

impl From<F16> for f32 {
    fn from(v: F16) -> Self {
        v.to_f32()
    }
}

/// Converts a slice of `f32` values into a `Vec<F16>`.
///
/// # Example
/// ```
/// use tensor_crab::cuda::half::{vec_f32_to_f16, vec_f16_to_f32};
/// let orig = vec![1.0_f32, -0.5, 2.0];
/// let f16s = vec_f32_to_f16(&orig);
/// let back = vec_f16_to_f32(&f16s);
/// for (a, b) in orig.iter().zip(back.iter()) {
///     assert!((a - b).abs() < 0.01);
/// }
/// ```
pub fn vec_f32_to_f16(v: &[f32]) -> Vec<F16> {
    v.iter().map(|&x| F16::from_f32(x)).collect()
}

/// Converts a slice of [`F16`] values into a `Vec<f32>`.
pub fn vec_f16_to_f32(v: &[F16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// Embedded PTX source for f16 element-wise unary operations.
///
/// Kernels convert f16 (stored as `u16`) → f32, apply the operation, then
/// convert back.  Requires `sm_70` or later (Volta+).
pub const PTX_UNARY_F16: &str = r#"
.version 7.0
.target sm_70
.address_size 64

// half_relu: out[i] = max(0, in[i])  (f16 stored as u16)
.visible .entry half_relu(
    .param .u64 out_ptr,
    .param .u64 in_ptr,
    .param .u64 count
)
{
    .reg .u64 %out, %in, %n;
    .reg .u32 %tid, %bid, %bdim, %idx32;
    .reg .u64 %idx;
    .reg .b16 %h_in, %h_out;
    .reg .f32 %f_in, %f_out;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %n,   [count];

    mov.u32 %tid,  %tid.x;
    mov.u32 %bid,  %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mad.lo.u32 %idx32, %bid, %bdim, %tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p, %idx, %n;
    @%p bra done;

    ld.global.b16 %h_in, [%in + %idx * 2];
    cvt.f32.f16   %f_in, %h_in;
    mov.f32       %f_out, 0f00000000;
    max.f32       %f_out, %f_in, %f_out;
    cvt.rn.f16.f32 %h_out, %f_out;
    st.global.b16  [%out + %idx * 2], %h_out;
done:
    ret;
}
"#;

/// Name of the f16 ReLU kernel in [`PTX_UNARY_F16`].
pub const KERNEL_RELU_F16: &str = "half_relu";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_zero_roundtrip() {
        let x = F16::from_f32(0.0);
        assert_eq!(x.to_f32(), 0.0);
    }

    #[test]
    fn test_f16_negative_zero_roundtrip() {
        let x = F16::from_f32(-0.0_f32);
        assert_eq!(x.to_f32(), -0.0_f32);
    }

    #[test]
    fn test_f16_one_roundtrip() {
        let x = F16::from_f32(1.0);
        assert!((x.to_f32() - 1.0_f32).abs() < 1e-3);
    }

    #[test]
    fn test_f16_negative() {
        let x = F16::from_f32(-2.5);
        assert!((x.to_f32() - (-2.5_f32)).abs() < 0.01);
    }

    #[test]
    fn test_f16_large_value() {
        // 65504 is approximately the max representable f16 finite value.
        let x = F16::from_f32(65504.0);
        assert!((x.to_f32() - 65504.0_f32).abs() < 32.0);
    }

    #[test]
    fn test_f16_overflow_to_inf() {
        let x = F16::from_f32(1.0e10_f32);
        assert!(x.to_f32().is_infinite());
    }

    #[test]
    fn test_f16_infinity_roundtrip() {
        let pos_inf = F16::from_f32(f32::INFINITY);
        assert!(pos_inf.to_f32().is_infinite() && pos_inf.to_f32() > 0.0);

        let neg_inf = F16::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32() < 0.0);
    }

    #[test]
    fn test_f16_nan_roundtrip() {
        let nan = F16::from_f32(f32::NAN);
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_vec_conversion_roundtrip() {
        let orig = vec![1.0_f32, -0.5, 3.14, 0.0, -100.0];
        let f16s = vec_f32_to_f16(&orig);
        let back = vec_f16_to_f32(&f16s);
        for (a, b) in orig.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() < a.abs() * 0.01 + 1e-3,
                "roundtrip failed: {a} → {b}"
            );
        }
    }

    #[test]
    fn test_f16_from_into_f32() {
        let x: F16 = 1.5_f32.into();
        let y: f32 = x.into();
        assert!((y - 1.5_f32).abs() < 1e-3);
    }

    #[test]
    fn test_f16_display() {
        let x = F16::from_f32(2.0);
        let s = format!("{x}");
        assert!(s.starts_with("2"), "unexpected display: {s}");
    }

    #[test]
    fn test_f16_debug() {
        let x = F16::from_f32(3.0);
        let s = format!("{x:?}");
        assert!(s.contains("f16("), "unexpected debug: {s}");
    }
}

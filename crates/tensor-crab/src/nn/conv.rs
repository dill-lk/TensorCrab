//! Convolutional and pooling layers.
//!
//! Provides 2-D convolution and pooling operations for image and sequence
//! processing.
//!
//! ## Layers
//!
//! | Type | Description |
//! |---|---|
//! | [`Conv2d`] | 2-D cross-correlation layer |
//! | [`MaxPool2d`] | 2-D max pooling |
//! | [`AvgPool2d`] | 2-D average pooling |

use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::autograd::graph::{BackwardFn, Node};
use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::Module;

// ─── Kernel initialisation ────────────────────────────────────────────────────

/// Kaiming uniform for convolutional kernels.
///
/// fan_in = channels_in * kH * kW
fn kaiming_conv(
    channels_out: usize,
    channels_in: usize,
    kh: usize,
    kw: usize,
    seed: u64,
) -> Tensor {
    let mut rng = SmallRng::seed_from_u64(seed);
    let fan_in = channels_in * kh * kw;
    let limit = (6.0_f32 / fan_in as f32).sqrt();
    let numel = channels_out * channels_in * kh * kw;
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-limit..=limit)).collect();
    Tensor::from_vec(data, &[channels_out, channels_in, kh, kw])
}

// ─── Forward convolution helpers ─────────────────────────────────────────────

/// Performs a 2-D cross-correlation (convolution without kernel flip).
///
/// - `input`: `[batch, C_in, H, W]`
/// - `weight`: `[C_out, C_in, kH, kW]`
/// - `bias`: optional `[C_out]`
/// - Returns: `[batch, C_out, H_out, W_out]`
fn conv2d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    padding: usize,
    stride: usize,
) -> Tensor {
    assert_eq!(input.ndim(), 4, "conv2d: input must be 4-D [B, C, H, W]");
    assert_eq!(
        weight.ndim(),
        4,
        "conv2d: weight must be 4-D [C_out, C_in, kH, kW]"
    );

    let batch = input.shape()[0];
    let c_in = input.shape()[1];
    let h_in = input.shape()[2];
    let w_in = input.shape()[3];

    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    assert_eq!(
        c_in,
        weight.shape()[1],
        "conv2d: input channels must match weight C_in"
    );

    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;

    let mut out_data = vec![0.0_f32; batch * c_out * h_out * w_out];

    for b in 0..batch {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0_f32;
                    for ci in 0..c_in {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride + khi;
                                let iw = ow * stride + kwi;
                                // Apply padding: treat out-of-bound as zero.
                                if ih >= padding
                                    && iw >= padding
                                    && ih < h_in + padding
                                    && iw < w_in + padding
                                {
                                    let src_ih = ih - padding;
                                    let src_iw = iw - padding;
                                    let inp = input.get_at(&[b, ci, src_ih, src_iw]);
                                    let w = weight.get_at(&[co, ci, khi, kwi]);
                                    acc += inp * w;
                                }
                            }
                        }
                    }
                    if let Some(b_tensor) = bias {
                        acc += b_tensor.get_at(&[co]);
                    }
                    let flat = b * (c_out * h_out * w_out) + co * (h_out * w_out) + oh * w_out + ow;
                    out_data[flat] = acc;
                }
            }
        }
    }

    Tensor::from_vec(out_data, &[batch, c_out, h_out, w_out])
}

// ─── Conv2d autograd backward ────────────────────────────────────────────────

struct Conv2dBackward {
    input_data: Tensor,
    weight_data: Tensor,
    has_bias: bool,
    padding: usize,
    stride: usize,
}

impl BackwardFn for Conv2dBackward {
    /// Returns gradients `[grad_input, grad_weight, (grad_bias)]`.
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let input = &self.input_data;
        let weight = &self.weight_data;

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];

        let c_out = weight.shape()[0];
        let kh = weight.shape()[2];
        let kw = weight.shape()[3];

        let h_out = grad_output.shape()[2];
        let w_out = grad_output.shape()[3];

        let p = self.padding;
        let s = self.stride;

        // ── grad w.r.t. input ──────────────────────────────────────────────
        let mut grad_input_data = vec![0.0_f32; batch * c_in * h_in * w_in];

        for b in 0..batch {
            for ci in 0..c_in {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        for co in 0..c_out {
                            let g = grad_output.get_at(&[b, co, oh, ow]);
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let ih = oh * s + khi;
                                    let iw = ow * s + kwi;
                                    if ih >= p && iw >= p && ih < h_in + p && iw < w_in + p {
                                        let src_ih = ih - p;
                                        let src_iw = iw - p;
                                        let flat = b * (c_in * h_in * w_in)
                                            + ci * (h_in * w_in)
                                            + src_ih * w_in
                                            + src_iw;
                                        grad_input_data[flat] +=
                                            g * weight.get_at(&[co, ci, khi, kwi]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input.shape());

        // ── grad w.r.t. weight ─────────────────────────────────────────────
        let mut grad_weight_data = vec![0.0_f32; c_out * c_in * kh * kw];

        for co in 0..c_out {
            for ci in 0..c_in {
                for khi in 0..kh {
                    for kwi in 0..kw {
                        let mut acc = 0.0_f32;
                        for b in 0..batch {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let ih = oh * s + khi;
                                    let iw = ow * s + kwi;
                                    if ih >= p && iw >= p && ih < h_in + p && iw < w_in + p {
                                        let src_ih = ih - p;
                                        let src_iw = iw - p;
                                        acc += input.get_at(&[b, ci, src_ih, src_iw])
                                            * grad_output.get_at(&[b, co, oh, ow]);
                                    }
                                }
                            }
                        }
                        let flat = co * (c_in * kh * kw) + ci * (kh * kw) + khi * kw + kwi;
                        grad_weight_data[flat] = acc;
                    }
                }
            }
        }

        let grad_weight = Tensor::from_vec(grad_weight_data, weight.shape());

        let mut grads = vec![grad_input, grad_weight];

        // ── grad w.r.t. bias ───────────────────────────────────────────────
        if self.has_bias {
            let mut grad_bias_data = vec![0.0_f32; c_out];
            for (co, grad_co) in grad_bias_data.iter_mut().enumerate() {
                for b in 0..batch {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            *grad_co += grad_output.get_at(&[b, co, oh, ow]);
                        }
                    }
                }
            }
            grads.push(Tensor::from_vec(grad_bias_data, &[c_out]));
        }

        grads
    }
}

// ─── Conv2d layer ─────────────────────────────────────────────────────────────

/// 2-D cross-correlation layer.
///
/// Applies a learnable set of filters to a 4-D input tensor of shape
/// `[batch, channels_in, height, width]`, producing an output of shape
/// `[batch, channels_out, height_out, width_out]`.
///
/// **Forward:** `output[b, co, oh, ow] = Σ(ci, kh, kw) input[b, ci, oh*s+kh, ow*s+kw] * weight[co, ci, kh, kw] + bias[co]`
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Conv2d};
///
/// let conv = Conv2d::new(1, 4, 3, 0, 1); // 1 in-channel, 4 out-channels, 3×3 kernel
/// let x = Variable::new(Tensor::randn_seeded(&[2, 1, 8, 8], 0), false);
/// let y = conv.forward(&x);
/// assert_eq!(y.data().shape(), &[2, 4, 6, 6]); // (8-3)/1+1 = 6
/// ```
pub struct Conv2d {
    /// Weight of shape `[channels_out, channels_in, kernel_h, kernel_w]`.
    pub weight: Arc<Variable>,
    /// Bias of shape `[channels_out]`.
    pub bias: Arc<Variable>,
    /// Number of input feature maps.
    pub channels_in: usize,
    /// Number of output feature maps.
    pub channels_out: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Padding applied to each spatial edge before the convolution.
    pub padding: usize,
    /// Convolution stride.
    pub stride: usize,
}

impl Conv2d {
    /// Creates a `Conv2d` layer with a square `kernel_size × kernel_size` kernel.
    ///
    /// Weights are initialised with Kaiming uniform.  Bias is zeroed.
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
    ) -> Self {
        Self::new_rectangular(
            channels_in,
            channels_out,
            kernel_size,
            kernel_size,
            padding,
            stride,
        )
    }

    /// Creates a `Conv2d` layer with a rectangular `kernel_h × kernel_w` kernel.
    pub fn new_rectangular(
        channels_in: usize,
        channels_out: usize,
        kernel_h: usize,
        kernel_w: usize,
        padding: usize,
        stride: usize,
    ) -> Self {
        let weight_data = kaiming_conv(channels_out, channels_in, kernel_h, kernel_w, 0);
        let bias_data = Tensor::zeros(&[channels_out]);
        Self {
            weight: Variable::new(weight_data, true),
            bias: Variable::new(bias_data, true),
            channels_in,
            channels_out,
            kernel_h,
            kernel_w,
            padding,
            stride,
        }
    }
}

impl Module for Conv2d {
    /// Forward pass: `[B, C_in, H, W]` → `[B, C_out, H_out, W_out]`.
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        let input_data = input.data().clone();
        let weight_data = self.weight.data().clone();
        let bias_data = self.bias.data().clone();

        let output_data = conv2d_forward(
            &input_data,
            &weight_data,
            Some(&bias_data),
            self.padding,
            self.stride,
        );

        let requires_grad =
            input.requires_grad || self.weight.requires_grad || self.bias.requires_grad;

        if !requires_grad {
            return Variable::new(output_data, false);
        }

        let node = Arc::new(Node {
            backward_fn: Box::new(Conv2dBackward {
                input_data,
                weight_data,
                has_bias: true,
                padding: self.padding,
                stride: self.stride,
            }),
            inputs: vec![
                Arc::clone(input),
                Arc::clone(&self.weight),
                Arc::clone(&self.bias),
            ],
        });

        Variable::with_grad_fn(output_data, true, node)
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![Arc::clone(&self.weight), Arc::clone(&self.bias)]
    }
}

// ─── MaxPool2d ────────────────────────────────────────────────────────────────

struct MaxPool2dBackward {
    input_data: Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
}

impl BackwardFn for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let input = &self.input_data;
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;
        let h_out = grad_output.shape()[2];
        let w_out = grad_output.shape()[3];

        let mut grad_input = vec![0.0_f32; batch * channels * h_in * w_in];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        // Find the position of the maximum in the pooling window.
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_ih = oh * s;
                        let mut max_iw = ow * s;

                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * s + khi;
                                let iw = ow * s + kwi;
                                if ih < h_in && iw < w_in {
                                    let v = input.get_at(&[b, c, ih, iw]);
                                    if v > max_val {
                                        max_val = v;
                                        max_ih = ih;
                                        max_iw = iw;
                                    }
                                }
                            }
                        }

                        // Only the max position receives the gradient.
                        let flat = b * (channels * h_in * w_in)
                            + c * (h_in * w_in)
                            + max_ih * w_in
                            + max_iw;
                        grad_input[flat] += grad_output.get_at(&[b, c, oh, ow]);
                    }
                }
            }
        }

        vec![Tensor::from_vec(grad_input, input.shape())]
    }
}

/// 2-D max pooling layer.
///
/// Slides a `kernel_h × kernel_w` window over the spatial dimensions and
/// keeps the maximum value in each window.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, MaxPool2d};
///
/// let pool = MaxPool2d::new(2, 2);
/// let x = Variable::new(Tensor::randn_seeded(&[1, 1, 4, 4], 0), false);
/// let y = pool.forward(&x);
/// assert_eq!(y.data().shape(), &[1, 1, 2, 2]);
/// ```
pub struct MaxPool2d {
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Pooling stride (defaults to kernel size when constructed with `new`).
    pub stride: usize,
}

impl MaxPool2d {
    /// Creates a `MaxPool2d` with a square kernel; stride equals `kernel_size`.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride,
        }
    }

    /// Creates a `MaxPool2d` with a rectangular kernel.
    pub fn new_rectangular(kernel_h: usize, kernel_w: usize, stride: usize) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride,
        }
    }
}

impl Module for MaxPool2d {
    /// Forward pass: `[B, C, H, W]` → `[B, C, H_out, W_out]`.
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        let input_data = input.data().clone();

        let batch = input_data.shape()[0];
        let channels = input_data.shape()[1];
        let h_in = input_data.shape()[2];
        let w_in = input_data.shape()[3];

        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;

        let h_out = (h_in - kh) / s + 1;
        let w_out = (w_in - kw) / s + 1;

        let mut out_data = vec![f32::NEG_INFINITY; batch * channels * h_out * w_out];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * s + khi;
                                let iw = ow * s + kwi;
                                if ih < h_in && iw < w_in {
                                    let v = input_data.get_at(&[b, c, ih, iw]);
                                    if v > max_val {
                                        max_val = v;
                                    }
                                }
                            }
                        }
                        let flat =
                            b * (channels * h_out * w_out) + c * (h_out * w_out) + oh * w_out + ow;
                        out_data[flat] = max_val;
                    }
                }
            }
        }

        let output_data = Tensor::from_vec(out_data, &[batch, channels, h_out, w_out]);

        if !input.requires_grad {
            return Variable::new(output_data, false);
        }

        let node = Arc::new(Node {
            backward_fn: Box::new(MaxPool2dBackward {
                input_data,
                kernel_h: kh,
                kernel_w: kw,
                stride: s,
            }),
            inputs: vec![Arc::clone(input)],
        });

        Variable::with_grad_fn(output_data, true, node)
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }
}

// ─── AvgPool2d ────────────────────────────────────────────────────────────────

struct AvgPool2dBackward {
    input_shape: Vec<usize>,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
}

impl BackwardFn for AvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let batch = self.input_shape[0];
        let channels = self.input_shape[1];
        let h_in = self.input_shape[2];
        let w_in = self.input_shape[3];
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;
        let h_out = grad_output.shape()[2];
        let w_out = grad_output.shape()[3];

        #[allow(clippy::cast_precision_loss)]
        let window_size = (kh * kw) as f32;
        let mut grad_input = vec![0.0_f32; batch * channels * h_in * w_in];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_output.get_at(&[b, c, oh, ow]) / window_size;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * s + khi;
                                let iw = ow * s + kwi;
                                if ih < h_in && iw < w_in {
                                    let flat = b * (channels * h_in * w_in)
                                        + c * (h_in * w_in)
                                        + ih * w_in
                                        + iw;
                                    grad_input[flat] += g;
                                }
                            }
                        }
                    }
                }
            }
        }

        vec![Tensor::from_vec(grad_input, &self.input_shape)]
    }
}

/// 2-D average pooling layer.
///
/// Slides a `kernel_h × kernel_w` window over the spatial dimensions and
/// computes the average value in each window.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, AvgPool2d};
///
/// let pool = AvgPool2d::new(2, 2);
/// let x = Variable::new(Tensor::ones(&[1, 1, 4, 4]), false);
/// let y = pool.forward(&x);
/// assert_eq!(y.data().shape(), &[1, 1, 2, 2]);
/// assert!((y.data().to_vec()[0] - 1.0).abs() < 1e-6); // avg of ones = 1
/// ```
pub struct AvgPool2d {
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Pooling stride.
    pub stride: usize,
}

impl AvgPool2d {
    /// Creates an `AvgPool2d` with a square kernel; stride equals `kernel_size`.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride,
        }
    }

    /// Creates an `AvgPool2d` with a rectangular kernel.
    pub fn new_rectangular(kernel_h: usize, kernel_w: usize, stride: usize) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride,
        }
    }
}

impl Module for AvgPool2d {
    /// Forward pass: `[B, C, H, W]` → `[B, C, H_out, W_out]`.
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        let input_data = input.data().clone();

        let batch = input_data.shape()[0];
        let channels = input_data.shape()[1];
        let h_in = input_data.shape()[2];
        let w_in = input_data.shape()[3];

        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;
        #[allow(clippy::cast_precision_loss)]
        let window_size = (kh * kw) as f32;

        let h_out = (h_in - kh) / s + 1;
        let w_out = (w_in - kw) / s + 1;

        let mut out_data = vec![0.0_f32; batch * channels * h_out * w_out];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0_f32;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * s + khi;
                                let iw = ow * s + kwi;
                                if ih < h_in && iw < w_in {
                                    sum += input_data.get_at(&[b, c, ih, iw]);
                                }
                            }
                        }
                        let flat =
                            b * (channels * h_out * w_out) + c * (h_out * w_out) + oh * w_out + ow;
                        out_data[flat] = sum / window_size;
                    }
                }
            }
        }

        let output_data = Tensor::from_vec(out_data, &[batch, channels, h_out, w_out]);

        if !input.requires_grad {
            return Variable::new(output_data, false);
        }

        let input_shape = input_data.shape().to_vec();
        let node = Arc::new(Node {
            backward_fn: Box::new(AvgPool2dBackward {
                input_shape,
                kernel_h: kh,
                kernel_w: kw,
                stride: s,
            }),
            inputs: vec![Arc::clone(input)],
        });

        Variable::with_grad_fn(output_data, true, node)
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use approx::assert_abs_diff_eq;

    // ── Conv2d forward ────────────────────────────────────────────────────

    #[test]
    fn test_conv2d_output_shape() {
        // (H_in - kH + 2*P) / S + 1 = (8 - 3 + 0) / 1 + 1 = 6
        let conv = Conv2d::new(1, 4, 3, 0, 1);
        let x = Variable::new(Tensor::randn_seeded(&[2, 1, 8, 8], 0), false);
        let y = conv.forward(&x);
        assert_eq!(y.data().shape(), &[2, 4, 6, 6]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // (8 - 3 + 2*1) / 1 + 1 = 8 (same convolution)
        let conv = Conv2d::new(3, 16, 3, 1, 1);
        let x = Variable::new(Tensor::randn_seeded(&[1, 3, 8, 8], 0), false);
        let y = conv.forward(&x);
        assert_eq!(y.data().shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_conv2d_stride() {
        // (8 - 3 + 0) / 2 + 1 = 3
        let conv = Conv2d::new(1, 1, 3, 0, 2);
        let x = Variable::new(Tensor::randn_seeded(&[1, 1, 8, 8], 0), false);
        let y = conv.forward(&x);
        assert_eq!(y.data().shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_known_values() {
        // 1-channel 1×1 conv on 2×2 input with stride 1, padding 0.
        // The kernel is the identity (single 1×1 kernel = 1.0), bias = 0.
        let mut conv = Conv2d::new(1, 1, 1, 0, 1);
        // Override weight to all-ones and bias to zero.
        conv.weight = Variable::new(Tensor::ones(&[1, 1, 1, 1]), true);
        conv.bias = Variable::new(Tensor::zeros(&[1]), true);

        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
            false,
        );
        let y = conv.forward(&x);
        assert_eq!(y.data().shape(), &[1, 1, 2, 2]);
        assert_abs_diff_eq!(
            y.data().to_vec().as_slice(),
            [1.0_f32, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_conv2d_backward_grad_exists() {
        let conv = Conv2d::new(1, 2, 3, 0, 1);
        let x = Variable::new(Tensor::randn_seeded(&[1, 1, 5, 5], 0), true);
        let y = conv.forward(&x);
        let loss = y.var_sum();
        backward(&loss);
        assert!(x.grad().is_some(), "Input gradient should be populated");
        assert!(
            conv.weight.grad().is_some(),
            "Weight gradient should be populated"
        );
        assert!(
            conv.bias.grad().is_some(),
            "Bias gradient should be populated"
        );
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 8, 3, 1, 1);
        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].data().shape(), &[8, 3, 3, 3]); // weight
        assert_eq!(params[1].data().shape(), &[8]); // bias
    }

    // ── MaxPool2d ─────────────────────────────────────────────────────────

    #[test]
    fn test_maxpool2d_output_shape() {
        let pool = MaxPool2d::new(2, 2);
        let x = Variable::new(Tensor::randn_seeded(&[2, 3, 8, 8], 0), false);
        let y = pool.forward(&x);
        assert_eq!(y.data().shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_known_values() {
        // [[1, 3, 2, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        // 2×2 max pool with stride 2: [[6, 8], [14, 16]]
        #[rustfmt::skip]
        let data = vec![
             1.0_f32,  3.0,  2.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let x = Variable::new(Tensor::from_vec(data, &[1, 1, 4, 4]), false);
        let pool = MaxPool2d::new(2, 2);
        let y = pool.forward(&x);
        assert_eq!(y.data().shape(), &[1, 1, 2, 2]);
        assert_abs_diff_eq!(
            y.data().to_vec().as_slice(),
            [6.0_f32, 8.0, 14.0, 16.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_maxpool2d_backward_grad_exists() {
        let pool = MaxPool2d::new(2, 2);
        let x = Variable::new(Tensor::randn_seeded(&[1, 1, 4, 4], 42), true);
        let y = pool.forward(&x);
        let loss = y.var_sum();
        backward(&loss);
        assert!(x.grad().is_some());
    }

    // ── AvgPool2d ─────────────────────────────────────────────────────────

    #[test]
    fn test_avgpool2d_output_shape() {
        let pool = AvgPool2d::new(2, 2);
        let x = Variable::new(Tensor::randn_seeded(&[2, 3, 8, 8], 0), false);
        let y = pool.forward(&x);
        assert_eq!(y.data().shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_avgpool2d_known_values() {
        // All ones → avg pool over 2×2 = 1.0 everywhere.
        let x = Variable::new(Tensor::ones(&[1, 1, 4, 4]), false);
        let pool = AvgPool2d::new(2, 2);
        let y = pool.forward(&x);
        assert_eq!(y.data().shape(), &[1, 1, 2, 2]);
        assert!(y.data().to_vec().iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_avgpool2d_backward_grad_exists() {
        let pool = AvgPool2d::new(2, 2);
        let x = Variable::new(Tensor::randn_seeded(&[1, 1, 4, 4], 42), true);
        let y = pool.forward(&x);
        let loss = y.var_sum();
        backward(&loss);
        assert!(x.grad().is_some());
        // Each element contributes equally: gradient should be 1/window_size = 0.25
        let g = x.grad().unwrap();
        assert!(g.to_vec().iter().all(|&v| (v - 0.25_f32).abs() < 1e-5));
    }
}

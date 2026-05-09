//! Linear regression example.
//!
//! Fits a line `y = 2x + 1 + noise` using gradient descent with MSE loss.
//! Demonstrates using raw autograd rather than the high-level `nn` module.
//!
//! Run with: `cargo run --example linear_regression`

use tensor_crab::autograd::{backward, Variable};
use tensor_crab::optim::{Optimizer, SGD};
use tensor_crab::tensor::Tensor;

fn main() {
    // ── Synthetic data: y = 2x + 1 + small noise ─────────────────────────────
    let n = 100usize;
    #[allow(clippy::cast_precision_loss)]
    let x_vals: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

    // Add small Gaussian noise via seeded randn.
    let noise = Tensor::randn_seeded(&[n], 42);
    let noise_scale = 0.05_f32;

    let y_vals: Vec<f32> = x_vals
        .iter()
        .zip(noise.to_vec().iter())
        .map(|(&xi, &ni)| 2.0 * xi + 1.0 + noise_scale * ni)
        .collect();

    let x_tensor = Tensor::from_vec(x_vals, &[n, 1]);
    let y_tensor = Tensor::from_vec(y_vals, &[n, 1]);

    // ── Model parameters: w (slope) and b (bias) ──────────────────────────────
    // Combine into a single weight matrix [w, b] applied to [x, 1] input.
    // Bias is handled by appending a column of ones to x.
    let ones = Tensor::ones(&[n, 1]);
    let x_with_bias =
        Tensor::cat(&[x_tensor, ones], 1).expect("cat: should not fail for [n, 1] and [n, 1]");

    // theta = [w, b] — single parameter vector of shape [2, 1]
    let theta = Variable::new(Tensor::zeros(&[2, 1]), true);

    // ── Optimiser ─────────────────────────────────────────────────────────────
    let mut opt = SGD::new(vec![theta.clone()], 0.5);

    // ── Training loop ─────────────────────────────────────────────────────────
    let x_var = Variable::new(x_with_bias, false);
    let y_var = Variable::new(y_tensor, false);

    println!("Training linear regression (y = 2x + 1)...");
    println!("Initial  theta = {:?}", theta.data().to_vec());

    for epoch in 0..=200 {
        // Forward: ŷ = X_aug @ theta  →  [n, 1]
        let pred = x_var.var_matmul(&theta);
        // MSE loss: mean((ŷ - y)²)
        let diff = pred.var_sub(&y_var);
        let loss = diff.var_mul(&diff).var_mean();

        backward(&loss);
        opt.step();
        opt.zero_grad();

        if epoch % 50 == 0 {
            let loss_val = loss.data().to_vec()[0];
            let t = theta.data().to_vec();
            println!(
                "Epoch {epoch:>3}  loss = {loss_val:.6}  w = {:.4}  b = {:.4}",
                t[0], t[1]
            );
        }
    }

    let t = theta.data().to_vec();
    println!(
        "\nFinal: w = {:.4} (target ≈ 2.0), b = {:.4} (target ≈ 1.0)",
        t[0], t[1]
    );
}

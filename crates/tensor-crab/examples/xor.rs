//! XOR problem — solved with a two-layer neural network.
//!
//! This example trains a small MLP to learn the XOR function:
//!
//! | x₁ | x₂ | y |
//! |---|---|---|
//! |  0 |  0 | 0 |
//! |  0 |  1 | 1 |
//! |  1 |  0 | 1 |
//! |  1 |  1 | 0 |
//!
//! Run with: `cargo run --example xor`

use tensor_crab::autograd::{backward, Variable};
use tensor_crab::nn::{loss, Linear, Module, ReLU, Sequential};
use tensor_crab::optim::{Adam, Optimizer};
use tensor_crab::tensor::Tensor;

fn main() {
    // ── Dataset ───────────────────────────────────────────────────────────────
    #[rustfmt::skip]
    let x_data = Tensor::from_vec(
        vec![
            0.0_f32, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ],
        &[4, 2],
    );
    let y_data = Tensor::from_vec(vec![0.0_f32, 1.0, 1.0, 0.0], &[4, 1]);

    let x = Variable::new(x_data, false);
    let y = Variable::new(y_data, false);

    // ── Model ─────────────────────────────────────────────────────────────────
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(8, 1)),
    ]);

    // ── Optimiser ─────────────────────────────────────────────────────────────
    let mut opt = Adam::new(model.parameters(), 0.01);

    // ── Training loop ─────────────────────────────────────────────────────────
    println!("Training XOR network...");
    for epoch in 0..=1000 {
        let pred = model.forward(&x);
        let l = loss::mse_loss(&pred, &y);
        backward(&l);
        opt.step();
        opt.zero_grad();

        if epoch % 100 == 0 {
            let loss_val = l.data().to_vec()[0];
            println!("Epoch {epoch:>4}  loss = {loss_val:.6}");
        }
    }

    // ── Evaluation ────────────────────────────────────────────────────────────
    println!("\nFinal predictions:");
    println!("  x₁  x₂  →  ŷ    (target)");
    let inputs = [[0.0_f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [0.0_f32, 1.0, 1.0, 0.0];
    for (inp, tgt) in inputs.iter().zip(targets.iter()) {
        let xi = Variable::new(Tensor::from_vec(inp.to_vec(), &[1, 2]), false);
        let pred = model.forward(&xi);
        let pred_val = pred.data().to_vec()[0];
        println!("  {}   {}   →  {:.3}  ({})", inp[0], inp[1], pred_val, tgt);
    }

    println!("\nDone. A well-trained network should output values close to 0/1.");
}

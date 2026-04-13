---
id: dataloader
title: DataLoader
sidebar_label: DataLoader
---

# DataLoader 📦

`DataLoader` batches and optionally shuffles a dataset each epoch, giving you clean mini-batches without boilerplate.

---

## Construction

```rust
use tensor_crab::optim::DataLoader;
use tensor_crab::tensor::Tensor;

// 1000 samples, 16 features each
let x = Tensor::from_vec(/* 16000 floats */, &[1000, 16]);
// 1000 labels
let y = Tensor::from_vec(/* 1000 floats */, &[1000, 1]);

let loader = DataLoader::new(
    x,          // inputs  — shape [n_samples, ...]
    y,          // targets — shape [n_samples, ...]
    32,         // batch_size
    true,       // shuffle each epoch
);
```

---

## Iterating Over Batches

Use `iter_epoch` inside a training loop. Pass a seed (usually the current epoch number) for reproducible shuffling, or `None` for random shuffling.

```rust
for epoch in 0..50 {
    for (x_batch, y_batch) in loader.iter_epoch(Some(epoch as u64)) {
        // x_batch: shape [32, 16]  (or smaller for the last batch)
        // y_batch: shape [32,  1]

        let x_var = Variable::new(x_batch, false);
        let y_var = Variable::new(y_batch, false);

        let pred = model.forward(&x_var);
        let l    = loss::mse_loss(&pred, &y_var);

        backward(&l);
        opt.step();
        opt.zero_grad();
    }
}
```

`iter_epoch` returns all batches in one epoch. The last batch may be smaller than `batch_size` if `n_samples` is not divisible by `batch_size`.

---

## Dataset Statistics

```rust
println!("{}", loader.n_samples()); // 1000
println!("{}", loader.n_batches()); // 32 (ceil(1000/32))
```

---

## Full Example

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::optim::{DataLoader, Optimizer, Adam};

fn main() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(4, 16)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(16, 1)),
    ]);

    let mut opt = Adam::new(model.parameters(), 0.001);

    // Synthetic dataset
    let x = Tensor::randn_seeded(&[256, 4], 42);
    let y = Tensor::zeros(&[256, 1]);

    let loader = DataLoader::new(x, y, 32, true);

    println!(
        "Dataset: {} samples, {} batches/epoch",
        loader.n_samples(),
        loader.n_batches()
    );

    for epoch in 0..20 {
        let mut epoch_loss = 0.0_f32;
        let mut n_batches  = 0_usize;

        for (x_batch, y_batch) in loader.iter_epoch(Some(epoch as u64)) {
            let x_var = Variable::new(x_batch, false);
            let y_var = Variable::new(y_batch, false);

            let pred = model.forward(&x_var);
            let l    = loss::mse_loss(&pred, &y_var);

            epoch_loss += l.data().to_vec()[0];
            n_batches  += 1;

            backward(&l);
            opt.step();
            opt.zero_grad();
        }

        println!(
            "Epoch {:>3} | avg loss: {:.4}",
            epoch,
            epoch_loss / n_batches as f32
        );
    }
}
```

---

## Notes

- **Shuffle seed**: passing `Some(epoch)` ensures each epoch has a different but reproducible shuffling, which is useful for debugging.
- **Thread safety**: `DataLoader` is `Send + Sync`, so you can share it across threads (though `iter_epoch` is single-threaded).
- **Memory**: batches are sliced from the original `Tensor` storage — no full copy of the dataset per epoch.

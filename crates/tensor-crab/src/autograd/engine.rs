use std::collections::HashSet;
use std::sync::Arc;

use crate::tensor::Tensor;

use super::Variable;

/// Runs the backward pass starting from `root`, populating `.grad()` on all
/// reachable leaf [`Variable`]s that have `requires_grad = true`.
///
/// `root` must be a scalar (numel == 1).  Its gradient is initialised to `1.0`
/// before propagation begins.
///
/// # Panics
/// Panics if `root.data().numel() != 1`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
///
/// let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
/// let y = x.var_mul(&x);   // y = x^2
/// let z = y.var_sum();      // z = sum(x^2) — scalar
/// backward(&z);
/// // dz/dx = 2x  =>  [4.0, 6.0]
/// let g = x.grad().unwrap();
/// assert!((g.to_vec()[0] - 4.0).abs() < 1e-5);
/// assert!((g.to_vec()[1] - 6.0).abs() < 1e-5);
/// ```
pub fn backward(root: &Arc<Variable>) {
    assert_eq!(
        root.data().numel(),
        1,
        "backward: root must be a scalar tensor (numel == 1), got shape {:?}",
        root.data().shape()
    );

    // Seed the gradient of the root with 1.0.
    {
        let seed = Tensor::ones(&[1]);
        root.accumulate_grad(&seed);
    }

    // Walk the graph in reverse topological order.
    let order = topo_sort(root);
    for var in order.iter().rev() {
        // Only propagate if this node has a backward function.
        if let Some(node) = &var.grad_fn {
            let grad_output = var.grad.lock().expect("backward: mutex poisoned").clone();

            if let Some(grad) = grad_output {
                let input_grads = node.backward_fn.backward(&grad);

                for (input_var, input_grad) in node.inputs.iter().zip(input_grads.iter()) {
                    if input_var.requires_grad {
                        input_var.accumulate_grad(input_grad);
                    }
                }
            }
        }
    }
}

// ─── Topological sort ────────────────────────────────────────────────────────

/// Returns the nodes in topological order (inputs before their output).
///
/// `backward` iterates this in *reverse* order so each node receives its
/// complete gradient before it propagates backward.
fn topo_sort(root: &Arc<Variable>) -> Vec<Arc<Variable>> {
    let mut visited: HashSet<*const Variable> = HashSet::new();
    let mut order: Vec<Arc<Variable>> = Vec::new();
    topo_visit(root, &mut visited, &mut order);
    order
}

fn topo_visit(
    var: &Arc<Variable>,
    visited: &mut HashSet<*const Variable>,
    order: &mut Vec<Arc<Variable>>,
) {
    let ptr = Arc::as_ptr(var);
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    if let Some(node) = &var.grad_fn {
        for input in &node.inputs {
            topo_visit(input, visited, order);
        }
    }

    order.push(Arc::clone(var));
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Numerical gradient checker ────────────────────────────────────────
    //
    // Computes the finite-difference gradient of `f` at `x` with step `eps`.
    // Used to verify that the autograd gradients are correct.

    fn numerical_grad(f: impl Fn(&Tensor) -> f32, x: &Tensor, eps: f32) -> Tensor {
        let n = x.numel();
        let mut grad_data = vec![0.0_f32; n];
        let x_data = x.to_vec();

        for i in 0..n {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps;
            xm[i] -= eps;

            let tp = Tensor::from_vec(xp, x.shape());
            let tm = Tensor::from_vec(xm, x.shape());
            grad_data[i] = (f(&tp) - f(&tm)) / (2.0 * eps);
        }

        Tensor::from_vec(grad_data, x.shape())
    }

    // ── zero_grad ─────────────────────────────────────────────────────────

    #[test]
    fn test_zero_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);
        let z = x.var_sum();
        backward(&z);
        assert!(x.grad().is_some());
        x.zero_grad();
        assert!(x.grad().is_none());
    }

    // ── add ───────────────────────────────────────────────────────────────

    #[test]
    fn test_add_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), true);
        let y = Variable::new(Tensor::from_vec(vec![4.0_f32, 5.0, 6.0], &[3]), true);
        let z = x.var_add(&y).var_sum();
        backward(&z);

        // dz/dx = 1 for all elements
        let gx = x.grad().unwrap();
        assert_abs_diff_eq!(
            gx.to_vec().as_slice(),
            [1.0_f32, 1.0, 1.0].as_slice(),
            epsilon = 1e-5
        );

        let gy = y.grad().unwrap();
        assert_abs_diff_eq!(
            gy.to_vec().as_slice(),
            [1.0_f32, 1.0, 1.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_add_numerical() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        let y_data = Tensor::from_vec(vec![4.0_f32, 5.0, 6.0], &[3]);

        let num = numerical_grad(
            |x| {
                let xv = Variable::new(x.clone(), false);
                let yv = Variable::new(y_data.clone(), false);
                xv.var_add(&yv).var_sum().data().to_vec()[0]
            },
            &data,
            1e-4,
        );

        let x = Variable::new(data, true);
        let y = Variable::new(y_data, false);
        let z = x.var_add(&y).var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── sub ───────────────────────────────────────────────────────────────

    #[test]
    fn test_sub_grad() {
        let x = Variable::new(Tensor::from_vec(vec![3.0_f32, 5.0], &[2]), true);
        let y = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);
        let z = x.var_sub(&y).var_sum();
        backward(&z);

        let gx = x.grad().unwrap();
        assert_abs_diff_eq!(
            gx.to_vec().as_slice(),
            [1.0_f32, 1.0].as_slice(),
            epsilon = 1e-5
        );

        let gy = y.grad().unwrap();
        assert_abs_diff_eq!(
            gy.to_vec().as_slice(),
            [-1.0_f32, -1.0].as_slice(),
            epsilon = 1e-5
        );
    }

    // ── mul ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mul_grad_x_squared() {
        // z = sum(x * x),  dz/dx = 2x
        let data = Tensor::from_vec(vec![2.0_f32, 3.0], &[2]);
        let x = Variable::new(data.clone(), true);
        let z = x.var_mul(&x).var_sum();
        backward(&z);

        let g = x.grad().unwrap();
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [4.0_f32, 6.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_mul_numerical() {
        let x_data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        let y_data = Tensor::from_vec(vec![4.0_f32, 5.0, 6.0], &[3]);

        let num = numerical_grad(
            |x| {
                let xv = Variable::new(x.clone(), false);
                let yv = Variable::new(y_data.clone(), false);
                xv.var_mul(&yv).var_sum().data().to_vec()[0]
            },
            &x_data,
            1e-4,
        );

        let x = Variable::new(x_data, true);
        let y = Variable::new(y_data, false);
        let z = x.var_mul(&y).var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── matmul ───────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_grad_numerical() {
        // Use small values to keep f32 numerical errors low.
        let x_data = Tensor::from_vec(vec![0.1_f32, 0.2, 0.3, 0.4], &[2, 2]);
        let y_data = Tensor::from_vec(vec![0.5_f32, 0.6, 0.7, 0.8], &[2, 2]);

        // Numerical grad w.r.t. x.
        let num_x = numerical_grad(
            |x| {
                let xv = Variable::new(x.clone(), false);
                let yv = Variable::new(y_data.clone(), false);
                xv.var_matmul(&yv).var_sum().data().to_vec()[0]
            },
            &x_data,
            1e-3,
        );

        let x = Variable::new(x_data.clone(), true);
        let y = Variable::new(y_data.clone(), true);
        let z = x.var_matmul(&y).var_sum();
        backward(&z);

        let gx = x.grad().unwrap();
        for (a, n) in gx.to_vec().iter().zip(num_x.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── neg ──────────────────────────────────────────────────────────────

    #[test]
    fn test_neg_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, -2.0, 3.0], &[3]), true);
        let z = x.var_neg().var_sum();
        backward(&z);

        let g = x.grad().unwrap();
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [-1.0_f32, -1.0, -1.0].as_slice(),
            epsilon = 1e-5
        );
    }

    // ── relu ─────────────────────────────────────────────────────────────

    #[test]
    fn test_relu_grad() {
        let x = Variable::new(Tensor::from_vec(vec![-1.0_f32, 0.5, 2.0], &[3]), true);
        let z = x.var_relu().var_sum();
        backward(&z);

        let g = x.grad().unwrap();
        // dReLU/dx = 1 if x > 0, else 0.  x=0 is edge case (treated as 0).
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [0.0_f32, 1.0, 1.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_relu_numerical() {
        let data = Tensor::from_vec(vec![1.5_f32, -0.5, 2.0], &[3]);
        let num = numerical_grad(
            |x| {
                Variable::new(x.clone(), false)
                    .var_relu()
                    .var_sum()
                    .data()
                    .to_vec()[0]
            },
            &data,
            1e-4,
        );

        let x = Variable::new(data, true);
        let z = x.var_relu().var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── sigmoid ──────────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_grad_numerical() {
        let data = Tensor::from_vec(vec![0.5_f32, -1.0, 2.0], &[3]);
        let num = numerical_grad(
            |x| {
                Variable::new(x.clone(), false)
                    .var_sigmoid()
                    .var_sum()
                    .data()
                    .to_vec()[0]
            },
            &data,
            1e-4,
        );

        let x = Variable::new(data, true);
        let z = x.var_sigmoid().var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── log ──────────────────────────────────────────────────────────────

    #[test]
    fn test_log_grad_numerical() {
        let data = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        let num = numerical_grad(
            |x| {
                Variable::new(x.clone(), false)
                    .var_log()
                    .var_sum()
                    .data()
                    .to_vec()[0]
            },
            &data,
            1e-4,
        );

        let x = Variable::new(data, true);
        let z = x.var_log().var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── exp ──────────────────────────────────────────────────────────────

    #[test]
    fn test_exp_grad_numerical() {
        let data = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3]);
        let num = numerical_grad(
            |x| {
                Variable::new(x.clone(), false)
                    .var_exp()
                    .var_sum()
                    .data()
                    .to_vec()[0]
            },
            &data,
            1e-4,
        );

        let x = Variable::new(data, true);
        let z = x.var_exp().var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    // ── sum ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sum_grad() {
        // z = sum(x),  dz/dx_i = 1 for all i.
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]), true);
        let z = x.var_sum();
        backward(&z);

        let g = x.grad().unwrap();
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [1.0_f32, 1.0, 1.0, 1.0].as_slice(),
            epsilon = 1e-5
        );
    }

    // ── chain rule / composed ops ─────────────────────────────────────────

    #[test]
    fn test_chain_add_mul() {
        // z = sum((x + y) * x) = sum(x^2 + xy)
        // dz/dx = 2x + y
        let x_data = Tensor::from_vec(vec![1.0_f32, 2.0], &[2]);
        let y_data = Tensor::from_vec(vec![3.0_f32, 4.0], &[2]);

        let num = numerical_grad(
            |x| {
                let xv = Variable::new(x.clone(), false);
                let yv = Variable::new(y_data.clone(), false);
                xv.var_add(&yv).var_mul(&xv).var_sum().data().to_vec()[0]
            },
            &x_data,
            1e-4,
        );

        let x = Variable::new(x_data, true);
        let y = Variable::new(y_data, false);
        let z = x.var_add(&y).var_mul(&x).var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }

    #[test]
    fn test_grad_accumulation_shared_input() {
        // When the same variable appears twice in the graph the gradients
        // must be accumulated (not overwritten).
        // z = sum(x + x),  dz/dx_i = 2.
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), true);
        let z = x.var_add(&x).var_sum();
        backward(&z);

        let g = x.grad().unwrap();
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [2.0_f32, 2.0, 2.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_no_grad_for_non_trainable() {
        // y does not require gradients — its .grad() should stay None.
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);
        let y = Variable::new(Tensor::from_vec(vec![3.0_f32, 4.0], &[2]), false);
        let z = x.var_mul(&y).var_sum();
        backward(&z);

        assert!(x.grad().is_some());
        assert!(y.grad().is_none());
    }

    #[test]
    fn test_sigmoid_relu_chain() {
        // z = sum(relu(sigmoid(x)))
        let data = Tensor::from_vec(vec![0.5_f32, -1.0, 2.0], &[3]);
        let num = numerical_grad(
            |x| {
                Variable::new(x.clone(), false)
                    .var_sigmoid()
                    .var_relu()
                    .var_sum()
                    .data()
                    .to_vec()[0]
            },
            &data,
            1e-4,
        );

        let x = Variable::new(data, true);
        let z = x.var_sigmoid().var_relu().var_sum();
        backward(&z);
        let analytic = x.grad().unwrap();

        for (a, n) in analytic.to_vec().iter().zip(num.to_vec().iter()) {
            assert_abs_diff_eq!(a, n, epsilon = 2e-2);
        }
    }
}

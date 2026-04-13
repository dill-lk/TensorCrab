import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

interface Feature {
  emoji: string;
  title: string;
  description: string;
}

const features: Feature[] = [
  {
    emoji: '🔒',
    title: 'Memory-Safe by Default',
    description:
      "Rust's borrow checker catches memory bugs at compile time. No leaks, no use-after-free, no data races — ever.",
  },
  {
    emoji: '⚡',
    title: 'Native Speed',
    description:
      'Zero interpreter overhead. Tensor operations compile directly to machine code — as fast as hand-written C++.',
  },
  {
    emoji: '🔁',
    title: 'Automatic Differentiation',
    description:
      'Dynamic computation graph built as you compute. Call backward() once and gradients flow everywhere.',
  },
  {
    emoji: '🧠',
    title: 'Composable NN Layers',
    description:
      'Linear, ReLU, Sigmoid, Tanh, Softmax, BatchNorm1d, Dropout — all implement Module and snap together.',
  },
  {
    emoji: '🚀',
    title: 'Production Optimizers',
    description:
      'SGD with momentum, Adam, AdamW, StepLR and CosineAnnealing schedulers — everything you need to train.',
  },
  {
    emoji: '📦',
    title: 'Single Binary Deploy',
    description:
      'No Python runtime. No pip. No venv. Your trained model ships as one self-contained binary.',
  },
];

const codeExample = `use tensor_crab::prelude::*;
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::autograd::{Variable, backward};

let model = Sequential::new(vec![
    Box::new(Linear::new(2, 16)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(16, 1)),
]);

let mut opt = Adam::new(model.parameters(), 0.001);

let x      = Variable::new(Tensor::from_vec(vec![0.0_f32, 1.0], &[1, 2]), false);
let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

let pred = model.forward(&x);
let l    = loss::mse_loss(&pred, &target);
backward(&l);   // gradients flow through the whole network
opt.step();
opt.zero_grad();`;

export default function Home(): ReactNode {
  return (
    <Layout
      title="TensorCrab — Rust ML Library"
      description="A blazing-fast ML library written entirely in Rust. No Python. No GIL. No overhead.">

      {/* ── Hero ── */}
      <section className="relative overflow-hidden border-b border-black/5 dark:border-white/10 bg-white dark:bg-[oklch(0.13_0.028_261.692)]">
        {/* Diagonal candy-cane gutters */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 left-0 w-10 hidden lg:block"
          style={{
            backgroundImage:
              'repeating-linear-gradient(315deg, var(--gutter-color) 0, var(--gutter-color) 1px, transparent 0, transparent 50%)',
            backgroundSize: '10px 10px',
            backgroundAttachment: 'fixed',
            // @ts-ignore
            '--gutter-color': 'rgba(0,0,0,0.05)',
          }}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 right-0 w-10 hidden lg:block dark:opacity-100"
          style={{
            backgroundImage:
              'repeating-linear-gradient(315deg, var(--gutter-color) 0, var(--gutter-color) 1px, transparent 0, transparent 50%)',
            backgroundSize: '10px 10px',
            backgroundAttachment: 'fixed',
            // @ts-ignore
            '--gutter-color': 'rgba(0,0,0,0.05)',
          }}
        />

        <div className="relative mx-auto max-w-5xl px-6 py-24 text-center lg:py-32">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/5 px-3 py-1 text-xs font-medium text-gray-600 dark:border-white/10 dark:bg-white/5 dark:text-gray-400">
            🦀 Pure Rust &nbsp;·&nbsp; No Python &nbsp;·&nbsp; No GIL
          </div>

          <Heading
            as="h1"
            className="mt-4 text-5xl font-bold tracking-tight text-gray-950 dark:text-white sm:text-6xl"
          >
            Machine learning,
            <br />
            <span className="text-sky-500 dark:text-sky-400">the Rust way.</span>
          </Heading>

          <p className="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            TensorCrab gives you N-dimensional tensors, automatic differentiation, composable
            layers, and gradient-based optimizers — all in a single Rust library with zero Python dependency.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              to="/docs/getting-started"
              className="rounded-lg bg-gray-950 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-gray-800 dark:bg-sky-500 dark:hover:bg-sky-400 dark:text-white no-underline transition-colors"
            >
              Get started →
            </Link>
            <Link
              to="/docs/intro"
              className="rounded-lg border border-black/10 dark:border-white/10 bg-white dark:bg-white/5 px-5 py-2.5 text-sm font-semibold text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-white/10 no-underline transition-colors"
            >
              What is TensorCrab?
            </Link>
          </div>

          {/* Install snippet */}
          <div className="mt-10 flex justify-center">
            <div className="inline-flex items-center gap-3 rounded-xl border border-black/10 dark:border-white/10 bg-gray-50 dark:bg-white/5 px-5 py-3">
              <span className="font-mono text-xs text-gray-500 dark:text-gray-400 select-all">
                tensor-crab = {'{'} git = "https://github.com/dill-lk/TensorCrab" {'}'}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features grid ── */}
      <section className="border-b border-black/5 dark:border-white/10 bg-gray-50/50 dark:bg-[oklch(0.165_0.025_260)]">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
            {features.map(({emoji, title, description}) => (
              <div key={title} className="rounded-xl border border-black/5 dark:border-white/10 bg-white dark:bg-white/5 p-6 shadow-sm">
                <div className="mb-3 text-2xl">{emoji}</div>
                <h3 className="mb-2 text-sm font-semibold text-gray-950 dark:text-white">{title}</h3>
                <p className="text-sm leading-7 text-gray-600 dark:text-gray-400">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Code example ── */}
      <section className="border-b border-black/5 dark:border-white/10 bg-white dark:bg-[oklch(0.13_0.028_261.692)]">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <p className="text-xs font-semibold tracking-widest uppercase text-sky-500 dark:text-sky-400 mb-3">
                Quick start
              </p>
              <Heading as="h2" className="text-3xl font-bold tracking-tight text-gray-950 dark:text-white mb-5">
                Train a model in 20 lines of Rust
              </Heading>
              <p className="text-sm leading-7 text-gray-600 dark:text-gray-400 mb-6">
                Build a network with <code>Sequential</code>, compute a loss, call{' '}
                <code>backward()</code> once, and let <code>Adam</code> update every weight.
                No boilerplate. No Python.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  to="/docs/getting-started"
                  className="text-sm font-semibold text-sky-600 dark:text-sky-400 hover:underline no-underline"
                >
                  Full guide →
                </Link>
                <Link
                  to="/docs/nn"
                  className="text-sm font-semibold text-gray-500 dark:text-gray-400 hover:underline no-underline"
                >
                  NN layers →
                </Link>
                <Link
                  to="/docs/optim"
                  className="text-sm font-semibold text-gray-500 dark:text-gray-400 hover:underline no-underline"
                >
                  Optimizers →
                </Link>
              </div>
            </div>

            {/* Code block */}
            <div className="overflow-hidden rounded-2xl bg-gray-950 shadow-2xl ring-1 ring-white/10">
              <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
                <div className="h-3 w-3 rounded-full bg-red-500/70" />
                <div className="h-3 w-3 rounded-full bg-yellow-500/70" />
                <div className="h-3 w-3 rounded-full bg-green-500/70" />
                <span className="ml-2 font-mono text-xs text-white/40">main.rs</span>
              </div>
              <pre className="overflow-x-auto p-5 text-xs leading-6 text-gray-300 m-0 rounded-none bg-transparent border-none">
                <code>{codeExample}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* ── Comparison table ── */}
      <section className="bg-gray-50/50 dark:bg-[oklch(0.165_0.025_260)]">
        <div className="mx-auto max-w-3xl px-6 py-20 text-center">
          <p className="text-xs font-semibold tracking-widest uppercase text-sky-500 dark:text-sky-400 mb-3">
            Why TensorCrab?
          </p>
          <Heading as="h2" className="text-3xl font-bold tracking-tight text-gray-950 dark:text-white mb-10">
            Less weight. More speed.
          </Heading>

          <div className="overflow-hidden rounded-xl border border-black/5 dark:border-white/10 bg-white dark:bg-white/5 shadow-sm text-left">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/5 dark:border-white/10">
                  <th className="py-3 px-5 font-semibold text-gray-950 dark:text-white" />
                  <th className="py-3 px-5 font-semibold text-gray-950 dark:text-white">Python (PyTorch)</th>
                  <th className="py-3 px-5 font-semibold text-gray-950 dark:text-white">🦀 TensorCrab</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Language', 'Python + C++', 'Pure Rust'],
                  ['Memory safety', 'Manual / GC', 'Compile-time ✓'],
                  ['Deployment', 'Heavy containers + Python', 'Single binary'],
                  ['Concurrency', 'GIL limits parallelism', 'True multi-threading'],
                  ['Overhead', 'Interpreter + FFI', 'Zero'],
                ].map(([feature, python, rust], i) => (
                  <tr
                    key={feature}
                    className={
                      i % 2 === 1
                        ? 'bg-gray-50 dark:bg-white/[0.02]'
                        : ''
                    }
                  >
                    <td className="py-3 px-5 font-medium text-gray-950 dark:text-white">{feature}</td>
                    <td className="py-3 px-5 text-gray-500 dark:text-gray-400">{python}</td>
                    <td className="py-3 px-5 text-sky-600 dark:text-sky-400 font-medium">{rust}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </Layout>
  );
}

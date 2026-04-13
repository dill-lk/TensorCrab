import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

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
      <section className={styles.heroSection}>
        <div aria-hidden="true" className={`${styles.heroGutter} ${styles.heroGutterLeft}`} />
        <div aria-hidden="true" className={`${styles.heroGutter} ${styles.heroGutterRight}`} />

        <div className={styles.heroInner}>
          <div className={styles.heroBadge}>
            🦀 Pure Rust &nbsp;·&nbsp; No Python &nbsp;·&nbsp; No GIL
          </div>

          <Heading as="h1" className={styles.heroTitle}>
            Machine learning,
            <br />
            <span className={styles.heroAccent}>the Rust way.</span>
          </Heading>

          <p className={styles.heroDesc}>
            TensorCrab gives you N-dimensional tensors, automatic differentiation, composable
            layers, and gradient-based optimizers — all in a single Rust library with zero Python dependency.
          </p>

          <div className={styles.heroButtons}>
            <Link to="/docs/getting-started" className={styles.btnPrimary}>
              Get started →
            </Link>
            <Link to="/docs/intro" className={styles.btnSecondary}>
              What is TensorCrab?
            </Link>
          </div>

          <div className={styles.installRow}>
            <div className={styles.installSnippet}>
              <span className={styles.installCode}>
                tensor-crab = {'{'} git = "https://github.com/dill-lk/TensorCrab" {'}'}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features grid ── */}
      <section className={styles.featuresSection}>
        <div className={styles.featuresInner}>
          <div className={styles.featuresGrid}>
            {features.map(({emoji, title, description}) => (
              <div key={title} className={styles.featureCard}>
                <div className={styles.featureEmoji}>{emoji}</div>
                <h3 className={styles.featureTitle}>{title}</h3>
                <p className={styles.featureDesc}>{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Code example ── */}
      <section className={styles.codeSection}>
        <div className={styles.codeInner}>
          <div className={styles.codeGrid}>
            <div>
              <p className={styles.codeLabel}>Quick start</p>
              <Heading as="h2" className={styles.codeTitle}>
                Train a model in 20 lines of Rust
              </Heading>
              <p className={styles.codeDesc}>
                Build a network with <code>Sequential</code>, compute a loss, call{' '}
                <code>backward()</code> once, and let <code>Adam</code> update every weight.
                No boilerplate. No Python.
              </p>
              <div className={styles.codeLinks}>
                <Link to="/docs/getting-started" className={styles.linkAccent}>
                  Full guide →
                </Link>
                <Link to="/docs/nn" className={styles.linkMuted}>
                  NN layers →
                </Link>
                <Link to="/docs/optim" className={styles.linkMuted}>
                  Optimizers →
                </Link>
              </div>
            </div>

            <div className={styles.codeBlockWrapper}>
              <div className={styles.codeBlockHeader}>
                <div className={`${styles.trafficDot} ${styles.trafficRed}`} />
                <div className={`${styles.trafficDot} ${styles.trafficYellow}`} />
                <div className={`${styles.trafficDot} ${styles.trafficGreen}`} />
                <span className={styles.codeBlockLabel}>main.rs</span>
              </div>
              <pre className={styles.codeBlockPre}>
                <code>{codeExample}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* ── Comparison table ── */}
      <section className={styles.comparisonSection}>
        <div className={styles.comparisonInner}>
          <p className={styles.comparisonLabel}>Why TensorCrab?</p>
          <Heading as="h2" className={styles.comparisonTitle}>
            Less weight. More speed.
          </Heading>

          <div className={styles.tableWrapper}>
            <table className={styles.comparisonTable}>
              <thead>
                <tr>
                  <th />
                  <th>Python (PyTorch)</th>
                  <th>🦀 TensorCrab</th>
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
                  <tr key={feature} className={i % 2 === 1 ? styles.tableRowAlt : ''}>
                    <td className={styles.tdFeature}>{feature}</td>
                    <td className={styles.tdPython}>{python}</td>
                    <td className={styles.tdRust}>{rust}</td>
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


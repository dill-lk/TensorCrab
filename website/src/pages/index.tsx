import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
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
      'Dynamic computation graph built implicitly as you compute. Call backward() once and gradients flow everywhere.',
  },
  {
    emoji: '🧠',
    title: 'Composable NN Layers',
    description:
      'Linear, ReLU, Sigmoid, Tanh, Softmax, BatchNorm1d, Dropout — all implement the Module trait and snap together.',
  },
  {
    emoji: '🚀',
    title: 'Production Optimizers',
    description:
      'SGD with momentum, Adam, AdamW with weight decay, StepLR and CosineAnnealing schedulers — everything you need to train.',
  },
  {
    emoji: '📦',
    title: 'Single Binary Deploy',
    description:
      'No Python runtime. No pip. No venv. Your trained model ships as one self-contained binary.',
  },
];

function FeatureCard({emoji, title, description}: Feature): ReactNode {
  return (
    <div className={clsx('col col--4', styles.featureCard)}>
      <div className={styles.featureEmoji}>{emoji}</div>
      <Heading as="h3">{title}</Heading>
      <p>{description}</p>
    </div>
  );
}

function HomepageHeader(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          TensorCrab 🦀
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started">
            Get Started →
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/docs/intro">
            What is TensorCrab?
          </Link>
        </div>
        <div className={styles.installBlock}>
          <code className={styles.installCode}>
            tensor-crab = {'{'} git = "https://github.com/dill-lk/TensorCrab" {'}'}
          </code>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="TensorCrab — Rust ML Library"
      description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {features.map((feat) => (
                <FeatureCard key={feat.title} {...feat} />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.codeSection}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              Train a network in 10 lines of Rust
            </Heading>
            <pre className={styles.codeBlock}>
              <code>{`use tensor_crab::prelude::*;
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::tensor::Tensor;

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
opt.step();     // update every weight
opt.zero_grad();`}</code>
            </pre>
            <div className={styles.ctaRow}>
              <Link className="button button--primary button--lg" to="/docs/getting-started">
                Read the full guide →
              </Link>
            </div>
          </div>
        </section>

        <section className={styles.comparisonSection}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              Why not PyTorch / TensorFlow?
            </Heading>
            <table className={styles.comparisonTable}>
              <thead>
                <tr>
                  <th></th>
                  <th>Python (PyTorch)</th>
                  <th>🦀 TensorCrab</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Language</strong></td>
                  <td>Python + C++</td>
                  <td>Pure Rust</td>
                </tr>
                <tr>
                  <td><strong>Memory safety</strong></td>
                  <td>Manual / GC</td>
                  <td>Compile-time</td>
                </tr>
                <tr>
                  <td><strong>Deployment</strong></td>
                  <td>Heavy containers + Python runtime</td>
                  <td>Single binary</td>
                </tr>
                <tr>
                  <td><strong>Concurrency</strong></td>
                  <td>GIL limits parallelism</td>
                  <td>True multi-threading</td>
                </tr>
                <tr>
                  <td><strong>Overhead</strong></td>
                  <td>Python interpreter + FFI</td>
                  <td>Zero</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </Layout>
  );
}

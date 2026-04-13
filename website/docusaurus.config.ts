import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'TensorCrab',
  tagline: 'A blazing-fast ML library written entirely in Rust. No Python. No GIL. No overhead.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://dill-lk.github.io',
  baseUrl: '/TensorCrab/',

  organizationName: 'dill-lk',
  projectName: 'TensorCrab',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/dill-lk/TensorCrab/edit/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/tensorcrab-social-card.png',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'TensorCrab 🦀',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api/tensor',
          label: 'API Reference',
          position: 'left',
        },
        {
          href: 'https://github.com/dill-lk/TensorCrab',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Getting Started', to: '/docs/getting-started'},
            {label: 'Tensor Engine', to: '/docs/tensor'},
            {label: 'Autograd', to: '/docs/autograd'},
            {label: 'Neural Networks', to: '/docs/nn'},
            {label: 'Optimizers', to: '/docs/optim'},
          ],
        },
        {
          title: 'API Reference',
          items: [
            {label: 'Tensor', to: '/docs/api/tensor'},
            {label: 'Variable', to: '/docs/api/variable'},
            {label: 'nn Modules', to: '/docs/api/nn'},
            {label: 'Optimizers', to: '/docs/api/optim'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'Roadmap', to: '/docs/roadmap'},
            {label: 'Architecture', to: '/docs/architecture'},
            {label: 'GitHub', href: 'https://github.com/dill-lk/TensorCrab'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TensorCrab. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.vsDark,
      additionalLanguages: ['rust', 'toml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

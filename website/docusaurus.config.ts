import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'TensorCrab',
  tagline: 'A blazing-fast ML library written entirely in Rust. No Python. No GIL. No overhead.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://tensor-crab.vercel.app',
  baseUrl: '/',

  organizationName: 'dill-lk',
  projectName: 'TensorCrab',

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  stylesheets: [
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..800;1,14..32,300..800&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap',
      type: 'text/css',
    },
  ],

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
      hideOnScroll: false,
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api/tensor',
          label: 'API',
          position: 'left',
        },
        {
          to: '/docs/roadmap',
          label: 'Roadmap',
          position: 'left',
        },
        {
          href: 'https://github.com/dill-lk/TensorCrab',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'light',
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
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.vsDark,
      additionalLanguages: ['rust', 'toml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

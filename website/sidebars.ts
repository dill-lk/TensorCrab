import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    {
      type: 'category',
      label: '🧮 Tensor Engine',
      items: ['tensor'],
    },
    {
      type: 'category',
      label: '🔁 Autograd',
      items: ['autograd'],
    },
    {
      type: 'category',
      label: '🧠 Neural Networks',
      items: ['nn'],
    },
    {
      type: 'category',
      label: '🚀 Optimizers',
      items: ['optim'],
    },
    'dataloader',
    {
      type: 'category',
      label: '📖 API Reference',
      items: [
        'api/tensor',
        'api/variable',
        'api/nn',
        'api/optim',
      ],
    },
    'architecture',
    'roadmap',
  ],
};

export default sidebars;

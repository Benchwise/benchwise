import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/core-concepts',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/evaluation',
        'guides/metrics',
        'guides/datasets',
        'guides/models',
        'guides/results',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: [
        'advanced/configuration',
        'advanced/custom-metrics',
        'advanced/error-handling',
        'advanced/logging',

      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        {
          type: 'category',
          label: 'Decorators',
          items: [
            'api/decorators/evaluate',
            'api/decorators/benchmark',
            'api/decorators/stress-test',
          ],
        },
        {
          type: 'category',
          label: 'Metrics',
          items: [
            'api/metrics/accuracy',
            'api/metrics/rouge',
            'api/metrics/bleu',
            'api/metrics/bert-score',
            'api/metrics/semantic-similarity',
            'api/metrics/safety',
            'api/metrics/coherence',
            'api/metrics/perplexity',
            'api/metrics/factual-correctness',
          ],
        },
        {
          type: 'category',
          label: 'Datasets',
          items: [
            'api/datasets/dataset',
            'api/datasets/load-dataset',
            'api/datasets/create-dataset',
          ],
        },
        {
          type: 'category',
          label: 'Models',
          items: [
            'api/models/model-adapter',
            'api/models/openai',
            'api/models/anthropic',
            'api/models/google',
            'api/models/huggingface',
          ],
        },
        {
          type: 'category',
          label: 'Results',
          items: [
            'api/results/evaluation-result',
            'api/results/benchmark-result',
            'api/results/results-analyzer',
          ],
        },
        // 'api/config',
        // 'api/client',
        'api/exceptions',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      items: [
        'examples/evaluation',
        'examples/datasets',
        'examples/results',
        'examples/metrics',
        'examples/question-answering',
        'examples/summarization',
        'examples/classification',
        'examples/safety-evaluation',
        'examples/multi-model-comparison',
      ],
    },
    {
      type: 'category',
      label: 'Additional',
      items: [
        'additional/cli',
        'additional/CHANGELOG',

        'additional/contributing',
        'additional/faq',
      ],
    },
  ],
};

export default sidebars;

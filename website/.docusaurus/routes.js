import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/benchwise/__docusaurus/debug',
    component: ComponentCreator('/benchwise/__docusaurus/debug', '482'),
    exact: true
  },
  {
    path: '/benchwise/__docusaurus/debug/config',
    component: ComponentCreator('/benchwise/__docusaurus/debug/config', '5da'),
    exact: true
  },
  {
    path: '/benchwise/__docusaurus/debug/content',
    component: ComponentCreator('/benchwise/__docusaurus/debug/content', '53c'),
    exact: true
  },
  {
    path: '/benchwise/__docusaurus/debug/globalData',
    component: ComponentCreator('/benchwise/__docusaurus/debug/globalData', '8bf'),
    exact: true
  },
  {
    path: '/benchwise/__docusaurus/debug/metadata',
    component: ComponentCreator('/benchwise/__docusaurus/debug/metadata', '6e7'),
    exact: true
  },
  {
    path: '/benchwise/__docusaurus/debug/registry',
    component: ComponentCreator('/benchwise/__docusaurus/debug/registry', 'fe0'),
    exact: true
  },
  {
    path: '/benchwise/__docusaurus/debug/routes',
    component: ComponentCreator('/benchwise/__docusaurus/debug/routes', '9f4'),
    exact: true
  },
  {
    path: '/benchwise/markdown-page',
    component: ComponentCreator('/benchwise/markdown-page', '0e2'),
    exact: true
  },
  {
    path: '/benchwise/docs',
    component: ComponentCreator('/benchwise/docs', 'd65'),
    routes: [
      {
        path: '/benchwise/docs',
        component: ComponentCreator('/benchwise/docs', '0b8'),
        routes: [
          {
            path: '/benchwise/docs',
            component: ComponentCreator('/benchwise/docs', '666'),
            routes: [
              {
                path: '/benchwise/docs/',
                component: ComponentCreator('/benchwise/docs/', 'd72'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/',
                component: ComponentCreator('/benchwise/docs/', '53e'),
                exact: true
              },
              {
                path: '/benchwise/docs/advanced/api-integration',
                component: ComponentCreator('/benchwise/docs/advanced/api-integration', '795'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/advanced/configuration',
                component: ComponentCreator('/benchwise/docs/advanced/configuration', '1e3'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/advanced/custom-metrics',
                component: ComponentCreator('/benchwise/docs/advanced/custom-metrics', 'b9b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/advanced/error-handling',
                component: ComponentCreator('/benchwise/docs/advanced/error-handling', '55b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/advanced/logging',
                component: ComponentCreator('/benchwise/docs/advanced/logging', 'f15'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/advanced/offline-mode',
                component: ComponentCreator('/benchwise/docs/advanced/offline-mode', '3fe'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/client',
                component: ComponentCreator('/benchwise/docs/api/client', '9b6'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/config',
                component: ComponentCreator('/benchwise/docs/api/config', '374'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/datasets/create-dataset',
                component: ComponentCreator('/benchwise/docs/api/datasets/create-dataset', 'b86'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/datasets/dataset',
                component: ComponentCreator('/benchwise/docs/api/datasets/dataset', 'b30'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/datasets/load-dataset',
                component: ComponentCreator('/benchwise/docs/api/datasets/load-dataset', '54a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/decorators/benchmark',
                component: ComponentCreator('/benchwise/docs/api/decorators/benchmark', '57d'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/decorators/evaluate',
                component: ComponentCreator('/benchwise/docs/api/decorators/evaluate', 'c92'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/decorators/stress-test',
                component: ComponentCreator('/benchwise/docs/api/decorators/stress-test', 'b14'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/exceptions',
                component: ComponentCreator('/benchwise/docs/api/exceptions', '79c'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/accuracy',
                component: ComponentCreator('/benchwise/docs/api/metrics/accuracy', '240'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/bert-score',
                component: ComponentCreator('/benchwise/docs/api/metrics/bert-score', 'a86'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/bleu',
                component: ComponentCreator('/benchwise/docs/api/metrics/bleu', '865'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/coherence',
                component: ComponentCreator('/benchwise/docs/api/metrics/coherence', '723'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/overview',
                component: ComponentCreator('/benchwise/docs/api/metrics/overview', '9d4'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/rouge',
                component: ComponentCreator('/benchwise/docs/api/metrics/rouge', 'bc7'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/safety',
                component: ComponentCreator('/benchwise/docs/api/metrics/safety', '94b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/metrics/semantic-similarity',
                component: ComponentCreator('/benchwise/docs/api/metrics/semantic-similarity', '13a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/models/anthropic',
                component: ComponentCreator('/benchwise/docs/api/models/anthropic', 'efd'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/models/google',
                component: ComponentCreator('/benchwise/docs/api/models/google', '980'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/models/huggingface',
                component: ComponentCreator('/benchwise/docs/api/models/huggingface', '5c3'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/models/model-adapter',
                component: ComponentCreator('/benchwise/docs/api/models/model-adapter', '85f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/models/openai',
                component: ComponentCreator('/benchwise/docs/api/models/openai', 'a02'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/overview',
                component: ComponentCreator('/benchwise/docs/api/overview', '82f'),
                exact: true
              },
              {
                path: '/benchwise/docs/api/results/benchmark-result',
                component: ComponentCreator('/benchwise/docs/api/results/benchmark-result', 'b4c'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/results/evaluation-result',
                component: ComponentCreator('/benchwise/docs/api/results/evaluation-result', '572'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/api/results/results-analyzer',
                component: ComponentCreator('/benchwise/docs/api/results/results-analyzer', 'df5'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/changelog',
                component: ComponentCreator('/benchwise/docs/changelog', 'dcb'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/cli',
                component: ComponentCreator('/benchwise/docs/cli', '390'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/contributing',
                component: ComponentCreator('/benchwise/docs/contributing', '293'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/examples',
                component: ComponentCreator('/benchwise/docs/examples', '521'),
                exact: true
              },
              {
                path: '/benchwise/docs/examples/classification',
                component: ComponentCreator('/benchwise/docs/examples/classification', 'c51'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/examples/multi-model-comparison',
                component: ComponentCreator('/benchwise/docs/examples/multi-model-comparison', '9bc'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/examples/question-answering',
                component: ComponentCreator('/benchwise/docs/examples/question-answering', '82a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/examples/safety-evaluation',
                component: ComponentCreator('/benchwise/docs/examples/safety-evaluation', 'f4b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/examples/summarization',
                component: ComponentCreator('/benchwise/docs/examples/summarization', '233'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/faq',
                component: ComponentCreator('/benchwise/docs/faq', 'bb8'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/getting-started',
                component: ComponentCreator('/benchwise/docs/getting-started', '1fd'),
                exact: true
              },
              {
                path: '/benchwise/docs/getting-started/core-concepts',
                component: ComponentCreator('/benchwise/docs/getting-started/core-concepts', 'eda'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/getting-started/installation',
                component: ComponentCreator('/benchwise/docs/getting-started/installation', 'e37'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/getting-started/quickstart',
                component: ComponentCreator('/benchwise/docs/getting-started/quickstart', 'a49'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/guides/datasets',
                component: ComponentCreator('/benchwise/docs/guides/datasets', '490'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/guides/evaluation',
                component: ComponentCreator('/benchwise/docs/guides/evaluation', '7f6'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/guides/metrics',
                component: ComponentCreator('/benchwise/docs/guides/metrics', '50e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/guides/models',
                component: ComponentCreator('/benchwise/docs/guides/models', 'cff'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/guides/results',
                component: ComponentCreator('/benchwise/docs/guides/results', '143'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/migration-guide',
                component: ComponentCreator('/benchwise/docs/migration-guide', 'fd6'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/benchwise/docs/usage-guide',
                component: ComponentCreator('/benchwise/docs/usage-guide', 'c50'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/benchwise/',
    component: ComponentCreator('/benchwise/', 'a98'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];

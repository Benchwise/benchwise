---
sidebar_position: 5
---

# FAQ

Frequently asked questions about Benchwise.

## General

### What is Benchwise?

Benchwise is an open-source Python SDK for LLM evaluation with PyTest-like syntax. It allows you to create custom evaluations, run benchmarks across multiple models, and share results with the community.

### Is Benchwise free?

Yes, Benchwise is open-source and free to use under the MIT license. You only pay for the LLM API calls to providers like OpenAI, Anthropic, etc.

### Which LLM providers are supported?

- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude 3 models)
- Google (Gemini)
- HuggingFace (any model)

## Usage

### How do I get started?

1. Install: `pip install benchwise`
2. Set API keys
3. Create your first evaluation

See the [Quickstart Guide](../getting-started/quickstart.md).

### Can I use custom metrics?

Yes! You can create custom metrics. See [Custom Metrics Guide](../advanced/custom-metrics.md).

### How do I compare multiple models?

Use the `@evaluate` decorator with multiple model names:

```python
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def my_test(model, dataset):
    # Test logic
    pass
```

### Can I run evaluations offline?

Yes, use offline mode to cache results locally. See [Offline Mode](../advanced/offline-mode.md).

## Technical

### Why async/await?

Async enables efficient concurrent API calls, reducing evaluation time when testing multiple models or large datasets.

### How are costs calculated?

Model adapters estimate costs based on token usage and provider pricing. Use `model.get_cost_estimate()` to check costs before running evaluations.

### Can I cache results?

Yes, results are automatically cached. Use `cache.clear_cache()` to clear when needed.

## Community

### How do I share benchmarks?

The community sharing platform is coming soon! You'll be able to upload and discover benchmarks.

### How can I contribute?

See the [Contributing Guide](./contributing.md).

### Where can I get help?

- [GitHub Issues](https://github.com/Benchwise/benchwise/issues)
- [Documentation](/)

## Troubleshooting

### API key errors?

Ensure API keys are set as environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Import errors?

Install optional dependencies:

```bash
pip install benchwise[all]
```

### Rate limiting?

Benchwise handles rate limits automatically. For heavy workloads, consider reducing concurrency or using smaller models.

## See Also

- [Getting Started](../getting-started/installation.md)
- [Guides](../guides/evaluation.md)
- [Examples](../examples/question-answering.md)

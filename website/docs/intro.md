---
sidebar_position: 1
slug: /
---

# Welcome to Benchwise

**Benchwise** is an open-source Python SDK for LLM evaluation with PyTest-like syntax. It allows you to create custom evaluations, run benchmarks across multiple models, and share results with the community.

## Why Benchwise?

- **PyTest-like Syntax** - Familiar decorator-based API (`@evaluate`, `@benchmark`)
- **Multi-Provider Support** - OpenAI, Anthropic, Google, HuggingFace
- **Built-in Metrics** - ROUGE, BLEU, BERT-score, semantic similarity, and more
- **Async-First** - Built for performance with async/await throughout
- **Results Management** - Caching, offline mode, and API upload
- **Dataset Tools** - Load standard benchmarks (MMLU, HellaSwag, GSM8K)
- **Community Sharing** - Share and discover benchmarks (Coming Soon)

## Quick Example

```python
from benchwise import evaluate, create_qa_dataset, accuracy
import asyncio

# Create a simple dataset
dataset = create_qa_dataset(
    questions=["What is the capital of France?", "What is 2+2?"],
    answers=["Paris", "4"]
)

# Evaluate multiple models
@evaluate("gpt-3.5-turbo", "claude-3-5-haiku-20241022")
async def test_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run it
results = asyncio.run(test_qa(dataset))
for result in results:
    print(f"{result.model_name}: {result.result['accuracy']:.2%}")
```

## Key Features

### Decorator-Based Evaluation

```python
@benchmark("medical_qa", "Medical question answering")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    return accuracy(responses, dataset.references)
```

### Multi-Provider Support

```python
# OpenAI models
@evaluate("gpt-4", "gpt-3.5-turbo")

# Anthropic models
@evaluate("claude-3-opus", "claude-3-sonnet")

# Google models
@evaluate("gemini-pro", "gemini-1.5-pro")

# HuggingFace models
@evaluate("microsoft/DialoGPT-medium")
```

### Built-in Metrics

```python
from benchwise.metrics import (
    rouge_l,           # Text overlap
    bleu_score,        # Translation quality
    bert_score_metric, # Semantic similarity
    accuracy,          # Exact match
    semantic_similarity, # Embedding similarity
    safety_score,      # Content safety
)
```

## Next Steps

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>Getting Started</h3>
      </div>
      <div className="card__body">
        <p>Learn how to install Benchwise and run your first evaluation.</p>
      </div>
      <div className="card__footer">
        <a href="/docs/getting-started" className="button button--primary button--block">Get Started</a>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>Usage Guide</h3>
      </div>
      <div className="card__body">
        <p>Explore detailed examples and best practices.</p>
      </div>
      <div className="card__footer">
        <a href="/docs/usage-guide" className="button button--secondary button--block">Learn More</a>
      </div>
    </div>
  </div>
</div>

## Community Features (Coming Soon)

We're building a platform to share and discover LLM evaluation benchmarks with the community:

- **Share Your Benchmarks** - Publish your evaluation results and benchmarks
- **Discover Benchmarks** - Browse community-contributed evaluations
- **Compare Results** - See how different models perform across various tasks
- **Leaderboards** - Track model performance across popular benchmarks

Stay tuned for updates on the community platform launch!

## Get Involved

- [GitHub Repository](https://github.com/Benchwise/benchwise) - Star us and contribute!
- [Issue Tracker](https://github.com/Benchwise/benchwise/issues) - Report bugs or request features
- [PyPI Package](https://pypi.org/project/benchwise/) - Install via pip

## License

Benchwise is open-source software licensed under the MIT license.

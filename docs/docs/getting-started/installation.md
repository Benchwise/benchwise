---
sidebar_position: 1
---

# Installation

Get started with Benchwise by installing the SDK and setting up your environment.

## Install via pip

```bash
pip install benchwise
```

## Install from Source

```bash
git clone https://github.com/Benchwise/benchwise.git
cd benchwise
pip install -e .
```

## Install with Optional Dependencies

Benchwise has several optional dependency groups for different use cases:

```bash
# Development tools
pip install -e ".[dev]"

# Testing tools
pip install -e ".[test]"

# Evaluation metrics
pip install -e ".[metrics]"

# LLM API clients
pip install -e ".[llm-apis]"

# HuggingFace transformers
pip install -e ".[transformers]"

# All LLM dependencies
pip install -e ".[llms]"

# Everything
pip install -e ".[all]"
```

## Set Up API Keys

To use LLM providers, you need to set up API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### Using a .env File

Alternatively, create a `.env` file in your project root:

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

## Verify Installation

Test that everything is installed correctly:

```python
import benchwise
print(benchwise.__version__)
```

## Next Steps

- [Quickstart Guide](./quickstart.md) - Run your first evaluation
- [Core Concepts](./core-concepts.md) - Understand key concepts

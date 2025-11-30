# Getting Started with Benchwise

Welcome to Benchwise! This guide will help you get up and running with evaluating LLMs using our platform.

## What is Benchwise?

Benchwise is an open-source library that makes LLM evaluation as easy as writing unit tests. With PyTest-like syntax, you can create custom evaluations, share benchmarks with the community, and monitor your models in production.

## Quick Start

### 1. Installation

```bash
# Install the SDK
pip install benchwise

# Or install from source
git clone https://github.com/your-org/benchwise.git
cd benchwise/sdk
pip install -e .
```

### 2. Set Up API Keys

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### 3. Your First Evaluation

Create a simple evaluation to test text summarization:

```python
from benchwise import evaluate, Dataset
from benchwise.metrics import rouge_l

# Create a simple dataset
data = [
    {
        "text": "The quick brown fox jumps over the lazy dog. This is a simple sentence.",
        "summary": "A fox jumps over a dog."
    }
]

dataset = Dataset("my_first_test", data)

@evaluate("gpt-3.5-turbo", "claude-3-haiku")
async def test_summarization(model, dataset):
    prompts = [f"Summarize: {item['text']}" for item in dataset.data]
    responses = await model.generate(prompts)
    references = [item['summary'] for item in dataset.data]
    
    scores = rouge_l(responses, references)
    assert scores['f1'] > 0.3  # Minimum quality threshold
    return scores

# Run the evaluation
import asyncio
results = asyncio.run(test_summarization(dataset))
print(results)
```



## Core Concepts

### Decorators

Benchwise uses decorators to make evaluation simple:

- `@evaluate(*models)` - Run tests on multiple models
- `@benchmark(name, description)` - Create named benchmarks
- `@stress_test(concurrent_requests, duration)` - Performance testing

### Models

Support for major LLM providers:

```python
# OpenAI models
@evaluate("gpt-4", "gpt-3.5-turbo")

# Anthropic models  
@evaluate("claude-3-opus", "claude-3-sonnet")

# Google models
@evaluate("gemini-pro", "gemini-1.5-pro")

# Hugging Face models
@evaluate("microsoft/DialoGPT-medium")
```

### Metrics

Built-in evaluation metrics:

```python
from benchwise.metrics import (
    rouge_l,           # Text overlap
    bleu_score,        # Translation quality
    bert_score_metric, # Semantic similarity
    accuracy,          # Exact match
    semantic_similarity, # Embedding similarity
    safety_score,      # Content safety
    coherence_score    # Text coherence
)
```

### Datasets

Create and manage datasets:

```python
from benchwise.datasets import Dataset, load_dataset

# Create custom dataset
dataset = Dataset(
    name="my_dataset",
    data=[{"input": "...", "output": "..."}],
    metadata={"task": "qa", "domain": "medical"}
)

# Load from file
dataset = load_dataset("path/to/data.json")
```

## Examples

### Question Answering

```python
@benchmark("medical_qa", "Medical question answering benchmark")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    questions = [f"Q: {item['question']}\nA:" for item in dataset.data]
    answers = await model.generate(questions, temperature=0)
    references = [item['answer'] for item in dataset.data]
    
    accuracy_score = accuracy(answers, references)
    similarity_score = semantic_similarity(answers, references)
    
    return {
        'accuracy': accuracy_score['accuracy'],
        'similarity': similarity_score['mean_similarity']
    }
```

### Safety Evaluation

```python
@benchmark("safety_check", "Evaluate model safety")
@evaluate("gpt-3.5-turbo", "claude-3-haiku")
async def test_safety(model, dataset):
    responses = await model.generate(dataset.prompts)
    
    safety_scores = safety_score(responses)
    assert safety_scores['mean_safety'] > 0.9  # High safety threshold
    
    return safety_scores
```

### Performance Testing

```python
@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def test_performance(model, dataset):
    start_time = time.time()
    response = await model.generate(["Hello, world!"])
    latency = time.time() - start_time
    
    assert latency < 2.0  # Max 2 second response time
    return {'latency': latency}
```


Happy evaluating! 🎯

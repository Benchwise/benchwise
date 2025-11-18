---
sidebar_position: 3
---

# Core Concepts

Understand the key concepts that power Benchwise.

## Decorators

Benchwise uses decorators to make evaluation simple and intuitive.

### @evaluate

The main decorator for running tests on multiple models:

```python
from benchwise import evaluate

# Single model
@evaluate("gpt-4")
async def test_single(model, dataset):
    pass

# Multiple models
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def test_multiple(model, dataset):
    pass

# With options
@evaluate("gpt-4", temperature=0.7, upload=True)
async def test_with_options(model, dataset):
    pass
```

### @benchmark

Mark evaluations as named benchmarks:

```python
from benchwise import benchmark, evaluate

@benchmark("Medical QA", "Evaluates medical question answering")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### @stress_test

Performance and load testing:

```python
from benchwise import stress_test, evaluate

@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def test_performance(model, dataset):
    response = await model.generate(["Hello, world!"])
    return {"response": response}
```

## Models

Benchwise supports multiple LLM providers out of the box.

### Supported Providers

```python
# OpenAI models
@evaluate("gpt-4", "gpt-3.5-turbo", "gpt-4-turbo")

# Anthropic models
@evaluate("claude-3-opus", "claude-3-sonnet", "claude-3-5-haiku-20241022")

# Google models
@evaluate("gemini-pro", "gemini-1.5-pro")

# HuggingFace models
@evaluate("microsoft/DialoGPT-medium")

# Mock adapter for testing
@evaluate("mock-test")
```

### Model Interface

All models provide a consistent async interface:

```python
async def my_test(model, dataset):
    # Generate text
    responses = await model.generate(prompts, temperature=0.7, max_tokens=100)

    # Get token count
    tokens = model.get_token_count(text)

    # Estimate cost
    cost = model.get_cost_estimate(input_tokens, output_tokens)
```

## Datasets

Datasets organize your evaluation data.

### Creating Datasets

```python
from benchwise import create_qa_dataset, create_summarization_dataset, load_dataset

# Question-Answer dataset
qa_data = create_qa_dataset(
    questions=["What is AI?", "Explain ML"],
    answers=["Artificial Intelligence", "Machine Learning"]
)

# Summarization dataset
summ_data = create_summarization_dataset(
    documents=["Long text here..."],
    summaries=["Summary here..."]
)

# Load from file
dataset = load_dataset("data.json")  # or .csv
```

### Dataset Properties

```python
# Access prompts and references
prompts = dataset.prompts
references = dataset.references

# Access raw data
data = dataset.data

# Dataset operations
sample = dataset.sample(n=10, random_state=42)
train, test = dataset.split(train_ratio=0.8)
filtered = dataset.filter(lambda x: len(x["question"]) > 10)
```

### Standard Benchmarks

```python
from benchwise import load_mmlu_sample, load_hellaswag_sample, load_gsm8k_sample

# Load benchmark samples
mmlu = load_mmlu_sample()
hellaswag = load_hellaswag_sample()
gsm8k = load_gsm8k_sample()
```

## Metrics

Built-in metrics for evaluating model outputs.

### Text Similarity

```python
from benchwise import rouge_l, bleu_score, bert_score_metric

# ROUGE-L for summarization
rouge = rouge_l(predictions, references)
print(rouge["f1"], rouge["precision"], rouge["recall"])

# BLEU for translation
bleu = bleu_score(predictions, references)

# BERT score for semantic similarity
bert = bert_score_metric(predictions, references)
```

### Accuracy & Correctness

```python
from benchwise import accuracy, factual_correctness

# Exact match accuracy
acc = accuracy(predictions, references)
print(acc["accuracy"])

# Factual correctness
correctness = factual_correctness(predictions, references)
```

### Semantic Similarity

```python
from benchwise import semantic_similarity, coherence_score

# Embedding-based similarity
similarity = semantic_similarity(predictions, references)
print(similarity["mean_similarity"])

# Text coherence
coherence = coherence_score(texts)
```

### Safety

```python
from benchwise import safety_score

# Content safety evaluation
safety = safety_score(responses)
print(safety["mean_safety"])
```

### Metric Collections

```python
from benchwise import get_text_generation_metrics, get_qa_metrics, get_safety_metrics

# Use predefined metric bundles
text_metrics = get_text_generation_metrics()
qa_metrics = get_qa_metrics()
safety_metrics = get_safety_metrics()

# Evaluate with multiple metrics
results = qa_metrics.evaluate(responses, references)
```

## Results

Handle and analyze evaluation results.

### EvaluationResult

Each model evaluation returns an `EvaluationResult`:

```python
result = results[0]
print(result.model_name)     # Model identifier
print(result.result)         # Your returned metrics
print(result.success)        # Whether evaluation succeeded
print(result.error)          # Error message if failed
print(result.duration)       # Time taken
print(result.metadata)       # Additional metadata
```

### BenchmarkResult

Organize multiple results:

```python
from benchwise import BenchmarkResult, save_results

benchmark = BenchmarkResult("My Benchmark")
for result in results:
    benchmark.add_result(result)

# Save in different formats
save_results(benchmark, "results.json", format="json")
save_results(benchmark, "results.csv", format="csv")
save_results(benchmark, "report.md", format="markdown")
```

### Analysis

```python
from benchwise import ResultsAnalyzer

# Generate reports
report = ResultsAnalyzer.generate_report(benchmark, "markdown")

# Compare models
comparison = benchmark.compare_models("accuracy")
print(f"Best: {comparison['best_model']}")
```

## Async-First Architecture

All evaluation functions are async:

```python
import asyncio

# Define async evaluation
@evaluate("gpt-4")
async def my_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

# Run from async code
async def main():
    results = await my_test(dataset)
    print(results)

asyncio.run(main())

# Or run directly
results = asyncio.run(my_test(dataset))
```

## Next Steps

- [Evaluation Guide](../guides/evaluation.md) - Learn evaluation patterns
- [Metrics Guide](../guides/metrics.md) - Deep dive into metrics
- [Datasets Guide](../guides/datasets.md) - Master dataset management

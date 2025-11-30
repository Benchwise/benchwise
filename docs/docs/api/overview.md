---
sidebar_position: 1
---

# Overview

Benchwise provides a comprehensive API for LLM evaluation. This section documents all public APIs.

## Core Modules

An overview of the main modules and their functionalities.

### `benchwise.core`

The main module containing evaluation decorators and orchestration.

- **`@evaluate(*models, **kwargs)`** - Main decorator for running tests on multiple models
- **`@benchmark(name, description, **metadata)`** - Decorator to mark tests as named benchmarks
- **`@stress_test(concurrent_requests, duration)`** - Performance testing decorator
- **`EvaluationRunner`** - Orchestrates evaluation execution

**Example:**
```python
from benchwise import evaluate, benchmark

@benchmark("qa_test", "Question answering evaluation")
@evaluate("gpt-4", "claude-3-opus")
async def test_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"results": responses}
```

### `benchwise.models`

Model adapters for different LLM providers.

- **`ModelAdapter`** - Abstract base class for all model adapters
- **`OpenAIAdapter`** - OpenAI API adapter (GPT models)
- **`AnthropicAdapter`** - Anthropic API adapter (Claude models)
- **`GoogleAdapter`** - Google Gemini API adapter
- **`HuggingFaceAdapter`** - HuggingFace models adapter
- **`MockAdapter`** - Mock adapter for testing
- **`get_model_adapter(model_name)`** - Factory function to get the appropriate adapter

**Example:**
```python
from benchwise.models import get_model_adapter

# Automatically selects the right adapter based on model name
adapter = get_model_adapter("gpt-4")
responses = await adapter.generate(["Hello, world!"])
```

### `benchwise.datasets`

Dataset management and loaders.

- **`Dataset`** - Main dataset class with smart property accessors
- **`load_dataset(path)`** - Load datasets from JSON/CSV files
- **`create_qa_dataset(questions, answers, **kwargs)`** - Create Q&A dataset
- **`create_summarization_dataset(documents, summaries, **kwargs)`** - Create summarization dataset
- **`create_classification_dataset(texts, labels, **kwargs)`** - Create classification dataset
- **`DatasetRegistry`** - Manage multiple datasets
- **`load_mmlu_sample()`** - Load MMLU benchmark sample
- **`load_hellaswag_sample()`** - Load HellaSwag benchmark sample
- **`load_gsm8k_sample()`** - Load GSM8K math benchmark sample

**Example:**
```python
from benchwise.datasets import create_qa_dataset, load_dataset

# Create custom dataset
dataset = create_qa_dataset(
    questions=["What is AI?"],
    answers=["Artificial Intelligence"]
)

# Load from file
dataset = load_dataset("my_data.json")

# Access data
prompts = dataset.prompts
references = dataset.references
```

### `benchwise.metrics`

Evaluation metrics for assessing model outputs.

**Text Similarity:**
- **`rouge_l(predictions, references)`** - ROUGE-L score
- **`bleu_score(predictions, references)`** - BLEU score
- **`bert_score_metric(predictions, references)`** - BERT-based semantic similarity

**Semantic:**
- **`semantic_similarity(predictions, references)`** - Embedding-based similarity
- **`coherence_score(texts)`** - Text coherence evaluation

**Evaluation:**
- **`accuracy(predictions, references)`** - Exact match accuracy
- **`factual_correctness(predictions, references, context)`** - Factual accuracy check

**Safety:**
- **`safety_score(texts)`** - Content safety evaluation

**Metric Collections:**
- **`MetricCollection`** - Bundle multiple metrics
- **`get_text_generation_metrics()`** - Common text generation metrics
- **`get_qa_metrics()`** - Q&A specific metrics
- **`get_safety_metrics()`** - Safety evaluation metrics

**Example:**
```python
from benchwise.metrics import rouge_l, accuracy, semantic_similarity

# Single metric
acc_result = accuracy(predictions, references)
print(f"Accuracy: {acc_result['accuracy']:.2%}")

# Multiple metrics
rouge_result = rouge_l(predictions, references)
sim_result = semantic_similarity(predictions, references)
```

### `benchwise.results`

Result management and analysis.

- **`EvaluationResult`** - Single model evaluation result
- **`BenchmarkResult`** - Collection of results across models
- **`ResultsAnalyzer`** - Statistical analysis and comparison
- **`ResultsCache`** - Local caching with JSON serialization
- **`save_results(results, path, format)`** - Save results to file
- **`load_results(path)`** - Load results from file

**Example:**
```python
from benchwise import save_results, BenchmarkResult, ResultsAnalyzer

# Create benchmark result
benchmark = BenchmarkResult("My Benchmark")
benchmark.add_result(result1)
benchmark.add_result(result2)

# Save in different formats
save_results(benchmark, "results.json", format="json")
save_results(benchmark, "results.csv", format="csv")
save_results(benchmark, "report.md", format="markdown")

# Analyze results
report = ResultsAnalyzer.generate_report(benchmark, "markdown")
print(report)
```

## Type Definitions

### `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    model_name: str
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `Dataset`

```python
@dataclass
class Dataset:
    name: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompts(self) -> List[str]: ...

    @property
    def references(self) -> List[str]: ...
```

## Next Steps

- [Getting Started](../guides/evaluation.md) - Learn how to use Benchwise
- [Examples](../examples/evaluation.md) - Practical examples

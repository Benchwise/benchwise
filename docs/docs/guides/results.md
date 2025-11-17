---
sidebar_position: 5
---

# Results

Learn how to manage, analyze, and export evaluation results.

## Understanding Results

### EvaluationResult

Each model evaluation returns an `EvaluationResult`:

```python
from benchwise import evaluate

@evaluate("gpt-4", "claude-3-opus")
async def my_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"accuracy": 0.85}

results = asyncio.run(my_test(dataset))

# Access result properties
for result in results:
    print(f"Model: {result.model_name}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Error: {result.error}")
    print(f"Metadata: {result.metadata}")
```

### Checking Success

```python
results = asyncio.run(my_evaluation(dataset))

successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

for failure in failed:
    print(f"Failed: {failure.model_name} - {failure.error}")
```

## Organizing Results

### BenchmarkResult

Organize multiple evaluations:

```python
from benchwise import BenchmarkResult, save_results

# Create benchmark result container
benchmark = BenchmarkResult(
    name="My Benchmark",
    metadata={"date": "2024-11-16", "version": "1.0"}
)

# Add individual results
results = asyncio.run(my_test(dataset))
for result in results:
    benchmark.add_result(result)

# Access results
all_results = benchmark.results
print(f"Total results: {len(all_results)}")
```

## Saving Results

### JSON Format

```python
from benchwise import save_results

save_results(benchmark, "results.json", format="json")
```

Example output:
```json
{
  "name": "My Benchmark",
  "metadata": {"date": "2024-11-16"},
  "results": [
    {
      "model_name": "gpt-4",
      "result": {"accuracy": 0.85},
      "success": true,
      "duration": 12.5
    }
  ]
}
```

### CSV Format

```python
save_results(benchmark, "results.csv", format="csv")
```

Example output:
```csv
model_name,accuracy,success,duration
gpt-4,0.85,true,12.5
claude-3-opus,0.82,true,11.2
```

### Markdown Report

```python
save_results(benchmark, "report.md", format="markdown")
```

Example output:
```markdown
# My Benchmark Results

| Model | Accuracy | Success | Duration |
|-------|----------|---------|----------|
| gpt-4 | 0.85 | ✓ | 12.5s |
| claude-3-opus | 0.82 | ✓ | 11.2s |
```

## Loading Results

```python
from benchwise import load_results

# Load previously saved results
benchmark = load_results("results.json")

print(f"Loaded: {benchmark.name}")
print(f"Results: {len(benchmark.results)}")
```

## Analyzing Results

### Compare Models

```python
# Find best performing model
comparison = benchmark.compare_models("accuracy")

print(f"Best model: {comparison['best_model']}")
print(f"Best score: {comparison['best_score']:.2%}")
print(f"Worst model: {comparison['worst_model']}")
print(f"Worst score: {comparison['worst_score']:.2%}")
```

### Generate Reports

```python
from benchwise import ResultsAnalyzer

# Generate markdown report
report = ResultsAnalyzer.generate_report(benchmark, format="markdown")
print(report)

# Generate HTML report
html_report = ResultsAnalyzer.generate_report(benchmark, format="html")

# Generate text report
text_report = ResultsAnalyzer.generate_report(benchmark, format="text")
```

### Statistical Analysis

```python
# Get summary statistics
stats = ResultsAnalyzer.get_statistics(benchmark, metric="accuracy")

print(f"Mean: {stats['mean']:.3f}")
print(f"Median: {stats['median']:.3f}")
print(f"Std Dev: {stats['std']:.3f}")
print(f"Min: {stats['min']:.3f}")
print(f"Max: {stats['max']:.3f}")
```

## Caching Results

Benchwise automatically caches results to avoid re-running expensive evaluations:

```python
from benchwise import cache

# Results are cached by default

# Clear cache when needed
cache.clear_cache()

# List cached results
cached = cache.list_cached_results()
print(f"Cached evaluations: {len(cached)}")

# Get specific cached result
cached_result = cache.get_cached_result("evaluation_id")
```

## Complete Example

```python
from benchwise import (
    evaluate,
    benchmark,
    create_qa_dataset,
    accuracy,
    semantic_similarity,
    save_results,
    BenchmarkResult,
    ResultsAnalyzer
)
import asyncio

# Create dataset
dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"]
)

# Run evaluation
@benchmark("AI Knowledge Test", "Tests understanding of AI concepts")
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def test_ai_knowledge(model, dataset):
    responses = await model.generate(dataset.prompts)

    acc = accuracy(responses, dataset.references)
    sim = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "similarity": sim["mean_similarity"]
    }

# Main execution
async def main():
    # Run evaluation
    results = await test_ai_knowledge(dataset)

    # Create benchmark result
    benchmark = BenchmarkResult(
        "AI Knowledge Benchmark",
        metadata={"date": "2024-11-16", "version": "1.0"}
    )

    for result in results:
        benchmark.add_result(result)

    # Save in multiple formats
    save_results(benchmark, "results.json", format="json")
    save_results(benchmark, "results.csv", format="csv")
    save_results(benchmark, "report.md", format="markdown")

    # Analyze
    comparison = benchmark.compare_models("accuracy")
    print(f"\nBest model: {comparison['best_model']}")
    print(f"Best accuracy: {comparison['best_score']:.2%}")

    # Generate report
    report = ResultsAnalyzer.generate_report(benchmark, "markdown")
    print("\n" + report)

    # Statistics
    stats = ResultsAnalyzer.get_statistics(benchmark, "accuracy")
    print(f"\nMean accuracy: {stats['mean']:.2%}")
    print(f"Std deviation: {stats['std']:.3f}")

asyncio.run(main())
```

## Best Practices

### 1. Always Save Results

```python
# Save after every major evaluation
save_results(benchmark, f"results_{timestamp}.json", format="json")
```

### 2. Include Metadata

```python
benchmark = BenchmarkResult(
    "My Benchmark",
    metadata={
        "date": "2024-11-16",
        "version": "2.0",
        "dataset_size": len(dataset.data),
        "models_tested": ["gpt-4", "claude-3-opus"],
        "environment": "production"
    }
)
```

### 3. Check for Failures

```python
results = asyncio.run(my_test(dataset))

failed = [r for r in results if not r.success]
if failed:
    print("WARNING: Some evaluations failed:")
    for f in failed:
        print(f"  - {f.model_name}: {f.error}")
```

### 4. Compare Over Time

```python
# Load previous results
old_results = load_results("results_2024_10.json")
new_results = load_results("results_2024_11.json")

# Compare improvements
old_best = old_results.compare_models("accuracy")["best_score"]
new_best = new_results.compare_models("accuracy")["best_score"]

improvement = new_best - old_best
print(f"Improvement: {improvement:+.2%}")
```

## Next Steps

- [Advanced Configuration](../advanced/configuration.md) - Configure Benchwise
- [API Integration](../advanced/api-integration.md) - Upload results to platform
- [API Reference](../api/results/evaluation-result.md) - Detailed results API

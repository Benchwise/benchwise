---
sidebar_position: 3
---

# Results Analyzer

Statistical analysis and reporting for results.

## Methods

### generate_report()
```python
@staticmethod
def generate_report(
    benchmark_result: BenchmarkResult,
    output_format: str = "text"
) -> str:
    ...
```

Generate a formatted report of benchmark results.

**Parameters:**
- **benchmark_result** (BenchmarkResult): Benchmark result to report on
- **output_format** (str): Format of the report. Options: `"text"`, `"markdown"`, `"html"`. Defaults to `"text"`.

**Returns:** Formatted report string

### compare_benchmarks()
```python
@staticmethod
def compare_benchmarks(
    benchmark_results: List[BenchmarkResult],
    metric_name: str = None
) -> Dict[str, Any]:
    ...
```

Compare results across multiple benchmarks.

**Parameters:**
- **benchmark_results** (List[BenchmarkResult]): List of benchmark results to compare
- **metric_name** (str, optional): Specific metric to compare

**Returns:** Dictionary with cross-benchmark comparison containing:
- `benchmarks`: List of benchmark information
- `models`: Set of all models across benchmarks
- `cross_benchmark_scores`: Model scores across benchmarks

### analyze_model_performance()
```python
@staticmethod
def analyze_model_performance(
    results: List[EvaluationResult],
    metric_name: str = None
) -> Dict[str, Any]:
    ...
```

Analyze performance of a single model across multiple evaluations.

**Parameters:**
- **results** (List[EvaluationResult]): List of evaluation results for the same model
- **metric_name** (str, optional): Specific metric to analyze

**Returns:** Dictionary with performance analysis containing:
- `model_name`: Name of the model
- `total_evaluations`: Total number of evaluations
- `successful_evaluations`: Number of successful evaluations
- `success_rate`: Rate of successful evaluations
- `mean_score`: Mean score across evaluations
- `median_score`: Median score
- `std_score`: Standard deviation of scores
- `min_score`: Minimum score
- `max_score`: Maximum score
- `score_range`: Range of scores (max - min)

## Usage

### Generate Reports

```python
from benchwise import ResultsAnalyzer

# Generate text report
text_report = ResultsAnalyzer.generate_report(benchmark, "text")
print(text_report)

# Generate markdown report
markdown_report = ResultsAnalyzer.generate_report(benchmark, "markdown")
with open("report.md", "w") as f:
    f.write(markdown_report)

# Generate HTML report
html_report = ResultsAnalyzer.generate_report(benchmark, "html")
with open("report.html", "w") as f:
    f.write(html_report)
```

### Analyze Model Performance

```python
# Collect all results for a specific model
gpt4_results = [r for r in all_results if r.model_name == "gpt-4"]

# Analyze performance
analysis = ResultsAnalyzer.analyze_model_performance(gpt4_results, "accuracy")
print(f"Model: {analysis['model_name']}")
print(f"Mean accuracy: {analysis['mean_score']:.3f}")
print(f"Median accuracy: {analysis['median_score']:.3f}")
print(f"Std Dev: {analysis['std_score']:.3f}")
print(f"Range: {analysis['min_score']:.3f} - {analysis['max_score']:.3f}")
print(f"Success rate: {analysis['success_rate']:.2%}")
```

### Compare Benchmarks

```python
from benchwise import load_results

# Load multiple benchmark results
benchmark1 = load_results("results_nov.json")
benchmark2 = load_results("results_dec.json")

# Compare across benchmarks
comparison = ResultsAnalyzer.compare_benchmarks(
    [benchmark1, benchmark2],
    metric_name="accuracy"
)

print(f"Models tested: {comparison['models']}")
print(f"\nCross-benchmark scores:")
for model, scores in comparison['cross_benchmark_scores'].items():
    print(f"{model}: {scores}")
```

## See Also

- [BenchmarkResult](./benchmark-result.md)
- [Results Guide](../../guides/results.md)

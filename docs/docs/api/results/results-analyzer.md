---
sidebar_position: 3
---

# Results Analyzer

Statistical analysis and reporting for results.

## Methods

### generate_report
```python
@staticmethod
def generate_report(
    benchmark: BenchmarkResult,
    format: str = "markdown"
) -> str
```

Generate formatted report. Formats: `markdown`, `html`, `text`.

### get_statistics
```python
@staticmethod
def get_statistics(
    benchmark: BenchmarkResult,
    metric: str
) -> Dict[str, float]
```

Calculate statistics for a metric.

## Usage

```python
from benchwise import ResultsAnalyzer

# Generate report
report = ResultsAnalyzer.generate_report(benchmark, "markdown")
print(report)

# Get statistics
stats = ResultsAnalyzer.get_statistics(benchmark, "accuracy")
print(f"Mean: {stats['mean']:.3f}")
print(f"Std Dev: {stats['std']:.3f}")
print(f"Min: {stats['min']:.3f}")
print(f"Max: {stats['max']:.3f}")
```

## See Also

- [BenchmarkResult](./benchmark-result.md)
- [Results Guide](../../guides/results.md)

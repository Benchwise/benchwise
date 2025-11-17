---
sidebar_position: 2
---

# Benchmark Result

Container for organizing multiple evaluation results.

## Class Definition

```python
class BenchmarkResult:
    def __init__(self, name: str, metadata: Dict[str, Any] = None):
        self.name = name
        self.metadata = metadata or {}
        self.results: List[EvaluationResult] = []
```

## Methods

### add_result
```python
def add_result(self, result: EvaluationResult) -> None
```

### compare_models
```python
def compare_models(self, metric: str) -> Dict[str, Any]
```

Returns best/worst model for given metric.

## Usage

```python
from benchwise import BenchmarkResult, save_results

benchmark = BenchmarkResult(
    "My Benchmark",
    metadata={"date": "2024-11-16"}
)

for result in results:
    benchmark.add_result(result)

# Compare
comparison = benchmark.compare_models("accuracy")
print(f"Best: {comparison['best_model']}")
print(f"Score: {comparison['best_score']}")

# Save
save_results(benchmark, "results.json", format="json")
```

## See Also

- [EvaluationResult](./evaluation-result.md)
- [save_results](../../guides/results.md)
- [Results Guide](../../guides/results.md)

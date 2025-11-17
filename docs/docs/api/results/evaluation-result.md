---
sidebar_position: 1
---

# Evaluation Result

Result from a single model evaluation.

## Class Definition

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

## Properties

- **model_name**: Model identifier
- **result**: Returned metrics dictionary
- **success**: Whether evaluation succeeded
- **error**: Error message if failed
- **duration**: Execution time in seconds
- **metadata**: Additional metadata

## Usage

```python
from benchwise import evaluate

@evaluate("gpt-4", "claude-3-opus")
async def my_test(model, dataset):
    return {"accuracy": 0.85}

results = asyncio.run(my_test(dataset))

for result in results:
    print(f"Model: {result.model_name}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Duration: {result.duration:.2f}s")
    if not result.success:
        print(f"Error: {result.error}")
```

## See Also

- [BenchmarkResult](./benchmark-result.md)
- [ResultsAnalyzer](./results-analyzer.md)
- [Results Guide](../../guides/results.md)

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
    test_name: str
    result: Any = None
    duration: float = 0.0
    dataset_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

## Attributes

- **model_name** (str): Name of the evaluated model
- **test_name** (str): Name of the test/benchmark that was run
- **result** (Any): The evaluation result (scores, metrics, etc.). Can be a dict, number, or any value
- **duration** (float): Time taken for evaluation in seconds
- **dataset_info** (Optional[Dict[str, Any]]): Information about the dataset used
- **error** (Optional[str]): Error message if evaluation failed
- **metadata** (Dict[str, Any]): Additional metadata about the evaluation
- **timestamp** (datetime): When the evaluation was completed

## Properties

- **success** (bool): Returns True if evaluation completed successfully (error is None)
- **failed** (bool): Returns True if evaluation failed (error is not None)

## Methods

### to_dict()
```python
def to_dict() -> Dict[str, Any]
```

Convert result to dictionary format for serialization.

**Returns:** Dictionary containing all result data including computed properties.

### get_score()
```python
def get_score(metric_name: str = None) -> Union[float, Any]
```

Extract a specific score from the result.

**Parameters:**
- **metric_name** (str, optional): Name of the metric to extract. If None, returns the main result.

**Returns:** The score value, or None if metric not found.

## Usage

```python
from benchwise import evaluate
import asyncio

@evaluate("gpt-4", "claude-3-opus")
async def my_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"accuracy": 0.85}

results = asyncio.run(my_test(dataset))

for result in results:
    print(f"Model: {result.model_name}")
    print(f"Test: {result.test_name}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Timestamp: {result.timestamp}")

    # Get specific metric
    accuracy = result.get_score("accuracy")
    print(f"Accuracy: {accuracy}")

    if result.failed:
        print(f"Error: {result.error}")
```

## See Also

- [BenchmarkResult](./benchmark-result.md)
- [ResultsAnalyzer](./results-analyzer.md)
- [Results Guide](../../guides/results.md)

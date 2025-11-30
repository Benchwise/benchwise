---
sidebar_position: 2
---

# Benchmark Result

Container for organizing multiple evaluation results.

## Class Definition

```python
@dataclass
class BenchmarkResult:
    benchmark_name: str
    results: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

## Attributes

- **benchmark_name** (str): Name of the benchmark
- **results** (List[EvaluationResult]): List of individual evaluation results
- **metadata** (Dict[str, Any]): Additional metadata about the benchmark run
- **timestamp** (datetime): When the benchmark was completed

## Properties

- **model_names** (List[str]): Get list of model names that were evaluated
- **successful_results** (List[EvaluationResult]): Get only successful evaluation results
- **failed_results** (List[EvaluationResult]): Get only failed evaluation results
- **success_rate** (float): Calculate the success rate of evaluations (0.0 to 1.0)

## Methods

### add_result()
```python
def add_result(self, result: EvaluationResult) -> None:
    ...
```

Add an evaluation result to the benchmark.

**Parameters:**
- **result** (EvaluationResult): The evaluation result to add

### compare_models()
```python
def compare_models(self, metric_name: str = None) -> Dict[str, Any]:
    ...
```

Compare all models in the benchmark.

**Parameters:**
- **metric_name** (str, optional): Specific metric to compare. If None, compares the main result.

**Returns:** Dictionary containing:
- `ranking`: List of models sorted by score (highest to lowest)
- `best_model`: Name of the best performing model
- `best_score`: Score of the best model
- `worst_model`: Name of the worst performing model
- `worst_score`: Score of the worst model
- `mean_score`: Mean score across all models
- `std_score`: Standard deviation of scores
- `total_models`: Number of models evaluated

### get_best_model()
```python
def get_best_model(self, metric_name: str = None) -> Optional[EvaluationResult]:
    ...
```

Get the best performing model result.

**Parameters:**
- **metric_name** (str, optional): Specific metric to compare. If None, compares the main result.

**Returns:** EvaluationResult of the best performing model, or None if no successful results.

### get_worst_model()
```python
def get_worst_model(self, metric_name: str = None) -> Optional[EvaluationResult]:
    ...
```

Get the worst performing model result.

**Parameters:**
- **metric_name** (str, optional): Specific metric to compare. If None, compares the main result.

**Returns:** EvaluationResult of the worst performing model, or None if no successful results.

### get_model_result()
```python
def get_model_result(self, model_name: str) -> Optional[EvaluationResult]:
    ...
```

Get result for a specific model.

**Parameters:**
- **model_name** (str): Name of the model to find

**Returns:** EvaluationResult for the model, or None if not found.

### to_dict()
```python
def to_dict(self) -> Dict[str, Any]:
    ...
```

Convert benchmark result to dictionary format for serialization.

**Returns:** Dictionary containing benchmark data and summary statistics.

### to_dataframe()
```python
def to_dataframe(self) -> pd.DataFrame:
    ...
```

Convert results to pandas DataFrame for analysis.

**Returns:** DataFrame with flattened result metrics.

### save_to_json()
```python
def save_to_json(self, file_path: Union[str, Path]) -> None:
    ...
```

Save benchmark results to JSON file.

**Parameters:**
- **file_path** (Union[str, Path]): Path where to save the JSON file

### save_to_csv()
```python
def save_to_csv(self, file_path: Union[str, Path]) -> None:
    ...
```

Save benchmark results to CSV file.

**Parameters:**
- **file_path** (Union[str, Path]): Path where to save the CSV file

## Usage

```python
from benchwise import BenchmarkResult, save_results

# Create benchmark result
benchmark = BenchmarkResult(
    benchmark_name="My Benchmark",
    metadata={"date": "2024-11-16", "version": "1.0"}
)

# Add results
for result in results:
    benchmark.add_result(result)

# Access properties
print(f"Models: {benchmark.model_names}")
print(f"Success rate: {benchmark.success_rate:.2%}")
print(f"Successful: {len(benchmark.successful_results)}")
print(f"Failed: {len(benchmark.failed_results)}")

# Compare models
comparison = benchmark.compare_models("accuracy")
print(f"Best: {comparison['best_model']}")
print(f"Score: {comparison['best_score']:.3f}")
print(f"Mean: {comparison['mean_score']:.3f}")

# Get specific model result
gpt4_result = benchmark.get_model_result("gpt-4")
if gpt4_result:
    print(f"GPT-4 accuracy: {gpt4_result.get_score('accuracy')}")

# Save using different methods
benchmark.save_to_json("results.json")
benchmark.save_to_csv("results.csv")
save_results(benchmark, "report.md", format="markdown")

# Convert to DataFrame for analysis
df = benchmark.to_dataframe()
print(df.describe())
```

## See Also

- [EvaluationResult](./evaluation-result.md)
- [save_results](../../guides/results.md)
- [Results Guide](../../guides/results.md)

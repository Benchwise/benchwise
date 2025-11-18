---
sidebar_position: 1
---

# Evaluate

The main decorator for running evaluations across multiple models.

## Signature

```python
@evaluate(*models: str, upload: bool = None, **kwargs) -> Callable
```

## Parameters

- **`*models`** (str): One or more model identifiers to evaluate
- **`upload`** (bool, optional): Whether to upload results to Benchwise API (None = use config default).
- **`**kwargs`**: Optional parameters passed to model generation:
  - `temperature` (float): Sampling temperature (0.0 to 1.0)
  - `max_tokens` (int): Maximum tokens to generate
  - `top_p` (float): Nucleus sampling parameter

## Returns

A decorator that wraps async functions to run evaluations across specified models.

## Basic Usage

```python
from benchwise import evaluate

@evaluate("gpt-4")
async def test_single_model(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Multiple Models

```python
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def test_multiple_models(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## With Parameters

```python
@evaluate("gpt-4", temperature=0.7, max_tokens=500)
async def test_with_params(model, dataset):
    # Temperature and max_tokens are applied to generation
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Function Signature

The decorated function must have this signature:

```python
async def evaluation_function(model: ModelAdapter, dataset: Dataset) -> Dict[str, Any]:
    # Your evaluation logic
    pass
```

### Parameters

- **`model`** (ModelAdapter): The model adapter instance for the current model
- **`dataset`** (Dataset): The dataset to evaluate on

### Returns

- **Dict[str, Any]**: Dictionary of metrics and results

## Model Interface

Inside the decorated function, the `model` parameter provides:

```python
# Generate text
responses = await model.generate(prompts, temperature=0.7, max_tokens=100)

# Get token count
tokens = model.get_token_count(text)

# Estimate cost
cost = model.get_cost_estimate(input_tokens, output_tokens)
```

## Execution

The decorator returns a list of `EvaluationResult` objects:

```python
results = asyncio.run(test_multiple_models(dataset))

for result in results:
    print(f"Model: {result.model_name}")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Duration: {result.duration}")
    if not result.success:
        print(f"Error: {result.error}")
```

## Complete Example

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

dataset = create_qa_dataset(
    questions=["What is AI?"],
    answers=["Artificial Intelligence"]
)

@evaluate("gpt-4", "claude-3-opus", temperature=0)
async def test_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total": len(responses)
    }

# Run evaluation
results = asyncio.run(test_qa(dataset))

# Process results
for result in results:
    if result.success:
        print(f"{result.model_name}: {result.result['accuracy']:.2%}")
```

## Combining with @benchmark

```python
from benchwise import benchmark, evaluate

@benchmark("QA Benchmark", "Question answering evaluation")
@evaluate("gpt-4", "claude-3-opus")
async def test_qa_benchmark(model, dataset):
    responses = await model.generate(dataset.prompts)
    return accuracy(responses, dataset.references)
```

## Error Handling

The decorator automatically handles errors:

```python
@evaluate("gpt-4", "invalid-model")
async def test_with_error(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

results = asyncio.run(test_with_error(dataset))

# Check for failures
for result in results:
    if not result.success:
        print(f"Error in {result.model_name}: {result.error}")
```

## Upload Results

Enable automatic upload to Benchwise API:

```python
@evaluate("gpt-4", upload=True)
async def test_with_upload(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## See Also

- [@benchmark](./benchmark.md) - Create named benchmarks
- [@stress_test](./stress-test.md) - Performance testing
- [Evaluation Guide](../../guides/evaluation.md) - Evaluation patterns

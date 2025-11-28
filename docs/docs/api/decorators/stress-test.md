---
sidebar_position: 3
---

# Stress Test

Decorator for performance and load testing with concurrent requests.

## Signature

```python
@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def stress_test_function(model, dataset):
    ...
```

## Parameters

- **`concurrent_requests`** (int): Number of concurrent requests to run
- **`duration`** (int): Duration in seconds to run the stress test

## Returns

A decorator that runs the evaluation function under load conditions.

## Basic Usage

```python
from benchwise import stress_test, evaluate

@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def test_performance(model, dataset):
    response = await model.generate(["Hello, world!"])
    return {"response": response}
```

## Complete Example

```python
import asyncio
import time
from benchwise import stress_test, evaluate, create_qa_dataset

dataset = create_qa_dataset(
    questions=["What is AI?"],
    answers=["Artificial Intelligence"]
)

@stress_test(concurrent_requests=5, duration=30)
@evaluate("gpt-3.5-turbo")
async def test_load(model, dataset):
    start_time = time.time()

    # Generate response
    response = await model.generate(dataset.prompts[:1])

    # Calculate latency
    latency = time.time() - start_time

    return {
        "latency": latency,
        "response_length": len(response[0]) if response else 0
    }

# Run stress test
results = asyncio.run(test_load(dataset))

# Analyze performance
for result in results:
    if result.success:
        print(f"Model: {result.model_name}")
        print(f"Latency: {result.result['latency']:.3f}s")
        print(f"Total Duration: {result.duration:.2f}s")
```

## Measuring Throughput

```python
@stress_test(concurrent_requests=20, duration=60)
@evaluate("gpt-3.5-turbo")
async def test_throughput(model, dataset):
    ...
```

## Latency Testing

```python
@stress_test(concurrent_requests=10, duration=30)
@evaluate("gpt-3.5-turbo", "gemini-2.5-flash")
async def test_latency(model, dataset):
    ...
```

## Testing Under Load

```python
@stress_test(concurrent_requests=50, duration=120)
@evaluate("gpt-3.5-turbo")
async def test_under_load(model, dataset):
    ...
```

## Rate Limiting Test

```python
@stress_test(concurrent_requests=100, duration=60)
@evaluate("gpt-3.5-turbo")
async def test_rate_limits(model, dataset):
    ...
```

## Cost Under Load

```python
@stress_test(concurrent_requests=20, duration=60)
@evaluate("gpt-3.5-turbo", "gemini-2.5-flash")
async def test_cost_under_load(model, dataset):
    ...
```

## Combining with @benchmark

```python
from benchwise import benchmark

@benchmark("Performance Benchmark", "Tests model performance under load")
@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo", "gemini-2.5-flash")
async def performance_benchmark(model, dataset):
    start = time.time()
    response = await model.generate(dataset.prompts[:1])
    latency = time.time() - start

    assert latency < 2.0, f"Latency {latency}s exceeds 2s threshold"

    return {"latency": latency}
```

## Best Practices

### 1. Start Small

```python
# Start with low concurrency
@stress_test(concurrent_requests=5, duration=30)
@evaluate("gpt-3.5-turbo")
async def gradual_test(model, dataset):
    ...

# Gradually increase
@stress_test(concurrent_requests=50, duration=60)
@evaluate("gpt-3.5-turbo")
async def heavy_test(model, dataset):
    ...
```

### 2. Monitor Costs

```python
@stress_test(concurrent_requests=10, duration=30)
@evaluate("gpt-3.5-turbo")
async def cost_aware_test(model, dataset):
    ...
```

### 3. Set Performance Thresholds

```python
@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def threshold_test(model, dataset):
    start = time.time()
    response = await model.generate(["Test"])
    latency = time.time() - start

    # Assert performance requirements
    assert latency < 2.0, f"Latency {latency}s exceeds threshold"

    return {"latency": latency, "passed": latency < 2.0}
```

## See Also

- [@evaluate](./evaluate.md) - Main evaluation decorator
- [@benchmark](./benchmark.md) - Create named benchmarks
- [Evaluation Guide](../../guides/evaluation.md) - Evaluation patterns

---
sidebar_position: 3
---

# Stress Test

Decorator for performance and load testing with concurrent requests.

## Signature

```python
@stress_test(concurrent_requests: int = 10, duration: int = 60) -> Callable
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
    start_time = time.time()
    responses = await model.generate(dataset.prompts)
    end_time = time.time()

    duration = end_time - start_time
    throughput = len(responses) / duration

    return {
        "throughput": throughput,  # requests per second
        "total_requests": len(responses),
        "duration": duration
    }

results = asyncio.run(test_throughput(dataset))
```

## Latency Testing

```python
@stress_test(concurrent_requests=10, duration=30)
@evaluate("gpt-4", "gpt-3.5-turbo")
async def test_latency(model, dataset):
    latencies = []

    for prompt in dataset.prompts:
        start = time.time()
        response = await model.generate([prompt])
        latency = time.time() - start
        latencies.append(latency)

    return {
        "avg_latency": sum(latencies) / len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)]
    }

results = asyncio.run(test_latency(dataset))

for result in results:
    if result.success:
        print(f"\n{result.model_name}:")
        print(f"  Avg Latency: {result.result['avg_latency']:.3f}s")
        print(f"  P95 Latency: {result.result['p95_latency']:.3f}s")
```

## Testing Under Load

```python
@stress_test(concurrent_requests=50, duration=120)
@evaluate("gpt-3.5-turbo")
async def test_under_load(model, dataset):
    """Test model performance under heavy load"""
    successful_requests = 0
    failed_requests = 0
    total_latency = 0

    for prompt in dataset.prompts:
        try:
            start = time.time()
            response = await model.generate([prompt])
            latency = time.time() - start

            successful_requests += 1
            total_latency += latency
        except Exception as e:
            failed_requests += 1

    avg_latency = total_latency / successful_requests if successful_requests > 0 else 0
    total_requests = successful_requests + failed_requests

    return {
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "failure_rate": failed_requests / total_requests if total_requests > 0 else 0,
        "avg_latency": avg_latency
    }

results = asyncio.run(test_under_load(dataset))
```

## Rate Limiting Test

```python
@stress_test(concurrent_requests=100, duration=60)
@evaluate("gpt-4")
async def test_rate_limits(model, dataset):
    """Test how the model handles rate limiting"""
    rate_limit_errors = 0
    successful_requests = 0

    for prompt in dataset.prompts[:10]:  # Test with smaller batch
        try:
            response = await model.generate([prompt])
            successful_requests += 1
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                rate_limit_errors += 1

    return {
        "successful_requests": successful_requests,
        "rate_limit_errors": rate_limit_errors,
        "handles_rate_limits": rate_limit_errors == 0
    }
```

## Cost Under Load

```python
@stress_test(concurrent_requests=20, duration=60)
@evaluate("gpt-4", "gpt-4o-mini")
async def test_cost_under_load(model, dataset):
    """Estimate costs under load conditions"""
    total_input_tokens = 0
    total_output_tokens = 0

    for prompt in dataset.prompts:
        input_tokens = model.get_token_count(prompt)
        response = await model.generate([prompt])

        if response:
            output_tokens = model.get_token_count(response[0])
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

    # get_cost_estimate returns total estimated cost in USD for the given token counts
    total_cost = model.get_cost_estimate(total_input_tokens, total_output_tokens)
    num_requests = len(dataset.prompts)

    return {
        "total_requests": num_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost": total_cost,
        "cost_per_request": total_cost / num_requests if num_requests > 0 else 0
    }

results = asyncio.run(test_cost_under_load(dataset))
```

## Combining with @benchmark

```python
from benchwise import benchmark

@benchmark("Performance Benchmark", "Tests model performance under load")
@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo", "claude-3-haiku")
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
@evaluate("gpt-4")
async def gradual_test(model, dataset):
    pass

# Gradually increase
@stress_test(concurrent_requests=50, duration=60)
@evaluate("gpt-4")
async def heavy_test(model, dataset):
    pass
```

### 2. Monitor Costs

```python
@stress_test(concurrent_requests=10, duration=30)
@evaluate("gpt-4")
async def cost_aware_test(model, dataset):
    # Estimate costs before running
    estimated_cost = model.get_cost_estimate(1000, 500) * 10 * 30  # Rough estimate

    if estimated_cost > 50:
        raise ValueError(f"Estimated cost ${estimated_cost:.2f} too high!")

    # Run test
    response = await model.generate(["Test"])
    return {"response": response}
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

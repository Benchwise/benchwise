---
sidebar_position: 1
---

# Evaluation

Learn how to create effective LLM evaluations with Benchwise.

## Basic Evaluation

The `@evaluate` decorator is your main tool for running evaluations:

```python
@evaluate("gpt-3.5-turbo")
async def test_basic(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

results = asyncio.run(test_basic(dataset))
```



## Multi-Model Comparison

Compare multiple models simultaneously:

```python
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def compare_models(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}
```



## Creating Benchmarks

Use `@benchmark` to create named, reusable evaluations:

```python
@benchmark("Medical QA v1.0", "Medical question answering evaluation")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }
```



## Model Configuration

Pass custom parameters to models:

```python
# High temperature for creative responses
@evaluate("gpt-4", temperature=0.7, max_tokens=500)
async def test_creative(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

# Temperature=0 for reproducible results
@evaluate("gpt-4", temperature=0)
async def test_deterministic(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```



## Error Handling

Benchwise automatically handles errors gracefully:

```python
@evaluate("gpt-4", "potentially-broken-model")
async def robust_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Check for failures
for result in results:
    if not result.success:
        print(f"Error in {result.model_name}: {result.error}")
```



## Custom Evaluation Logic

You have full control over evaluation logic:

```python
@evaluate("gpt-4")
async def custom_evaluation(model, dataset):
    responses = []
    for prompt in dataset.prompts:
        # Custom prompt engineering
        enhanced_prompt = f"Answer concisely: {prompt}"
        response = await model.generate([enhanced_prompt], temperature=0.5)
        responses.extend(response)

    # Custom scoring logic
    scores = [len(r) > 0 for r in responses]
    return {
        "custom_score": sum(scores) / len(scores),
        "total": len(scores)
    }
```



## Batch Processing

Handle large datasets efficiently:

```python
@evaluate("gpt-3.5-turbo")
async def batch_evaluation(model, dataset):
    batch_size = 10
    all_responses = []

    # Process in batches
    for i in range(0, len(dataset.prompts), batch_size):
        batch = dataset.prompts[i:i+batch_size]
        responses = await model.generate(batch)
        all_responses.extend(responses)

    scores = accuracy(all_responses, dataset.references)
    return {"accuracy": scores["accuracy"]}
```



## Result Upload

Enable automatic result upload (when platform is available):

```python
@evaluate("gpt-4", upload=True)
async def test_with_upload(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}
```



## Saving Results

Save evaluation results for later analysis:

```python
async def run_and_save():
    results = await test_medical_qa(dataset)

    # Create benchmark result container
    benchmark = BenchmarkResult("Medical QA Results")
    for result in results:
        benchmark.add_result(result)

    # Save in multiple formats
    save_results(benchmark, "results.json", format="json")
    save_results(benchmark, "results.csv", format="csv")
    save_results(benchmark, "report.md", format="markdown")

asyncio.run(run_and_save())
```



## Best Practices

### 1. Use Descriptive Names

Give your benchmarks clear, versioned names:

```python
@benchmark("customer_support_qa_v2", "Customer support QA - Updated 2024")
@evaluate("gpt-4")
async def test_customer_support(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}
```

### 2. Set Temperature Appropriately

```python
# For factual tasks - use temperature=0
@evaluate("gpt-4", temperature=0)
async def test_facts(model, dataset):
    responses = await model.generate(dataset.prompts)
    return accuracy(responses, dataset.references)

# For creative tasks - use higher temperature
@evaluate("gpt-4", temperature=0.9)
async def test_creative(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"response": responses[0]}
```

### 3. Return Comprehensive Metrics

Track multiple metrics beyond just accuracy:

```python
@evaluate("gpt-4")
async def comprehensive_test(model, dataset):
    start_time = time.time()
    responses = await model.generate(dataset.prompts)
    duration = time.time() - start_time

    return {
        "accuracy": accuracy(responses, dataset.references)["accuracy"],
        "total_samples": len(responses),
        "avg_length": sum(len(r) for r in responses) / len(responses),
        "duration": duration
    }
```

### 4. Test with Samples First

Test on a small sample before running on the full dataset:

```python
# Test with small sample first
test_sample = full_dataset.sample(n=10, random_state=42)
results = asyncio.run(my_test(test_sample))

# Then run on full dataset
results = asyncio.run(my_test(full_dataset))
```

:::info Complete Examples
For full runnable code with imports, execution, and various scenarios demonstrating evaluation patterns, see [Evaluation](../examples/evaluation.md).
:::

## Next Steps

- [Metrics Guide](./metrics.md) - Learn about evaluation metrics
- [Datasets Guide](./datasets.md) - Master dataset management
- [Models Guide](./models.md) - Understand model adapters

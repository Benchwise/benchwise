---
sidebar_position: 1
---

# Evaluation

Learn how to create effective LLM evaluations with Benchwise.

## Basic Evaluation

The `@evaluate` decorator is your main tool for running evaluations:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

dataset = create_qa_dataset(
    questions=["What is the capital of France?"],
    answers=["Paris"]
)

@evaluate("gpt-3.5-turbo")
async def test_basic(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run the evaluation
results = asyncio.run(test_basic(dataset))

# Print results
for result in results:
    if result.success:
        print(f"{result.model_name}: Accuracy = {result.result['accuracy']:.2%}")
    else:
        print(f"{result.model_name}: FAILED - {result.error}")
```

## Multi-Model Comparison

Compare multiple models simultaneously:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset
dataset = create_qa_dataset(
    questions=["What is the capital of France?", "What is 2+2?"],
    answers=["Paris", "4"]
)

@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def compare_models(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run the evaluation
results = asyncio.run(compare_models(dataset))

# Results is a list of EvaluationResult objects
for result in results:
    if result.success:
        print(f"{result.model_name}: {result.result['accuracy']:.2%}")
    else:
        print(f"{result.model_name}: FAILED - {result.error}")
```

## Creating Benchmarks

Use `@benchmark` to create named, reusable evaluations:

```python
import asyncio
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy

# Create medical QA dataset
dataset = create_qa_dataset(
    questions=["What is aspirin used for?", "What is the normal body temperature?"],
    answers=["Pain relief and reducing fever", "98.6°F or 37°C"]
)

@benchmark("Medical QA v1.0", "Medical question answering evaluation")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }

# Run the benchmark
results = asyncio.run(test_medical_qa(dataset))

# Print results
for result in results:
    if result.success:
        print(f"{result.model_name}:")
        print(f"  Accuracy: {result.result['accuracy']:.2%}")
        print(f"  Total Questions: {result.result['total_questions']}")
    else:
        print(f"{result.model_name}: FAILED - {result.error}")
```

## Model Configuration

Pass custom parameters to models:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset

# Create dataset
dataset = create_qa_dataset(
    questions=["Write a creative story about AI"],
    answers=["A story about artificial intelligence"]
)

@evaluate("gpt-4", temperature=0.7, max_tokens=500)
async def test_creative(model, dataset):
    # High temperature for creative responses
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

@evaluate("gpt-4", temperature=0)
async def test_deterministic(model, dataset):
    # Temperature=0 for reproducible results
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

# Run both evaluations
creative_results = asyncio.run(test_creative(dataset))
deterministic_results = asyncio.run(test_deterministic(dataset))

print("Creative response:")
for result in creative_results:
    if result.success:
        print(f"{result.model_name}: {result.result['responses'][0][:100]}...")

print("\nDeterministic response:")
for result in deterministic_results:
    if result.success:
        print(f"{result.model_name}: {result.result['responses'][0][:100]}...")
```

## Error Handling

Benchwise automatically handles errors gracefully:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset
dataset = create_qa_dataset(
    questions=["What is AI?"],
    answers=["Artificial Intelligence"]
)

@evaluate("gpt-4", "potentially-broken-model")
async def robust_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run the evaluation
results = asyncio.run(robust_test(dataset))

# Check for failures
for result in results:
    if result.success:
        print(f"{result.model_name}: Accuracy = {result.result['accuracy']:.2%}")
    else:
        print(f"Error in {result.model_name}: {result.error}")
```

## Custom Evaluation Logic

You have full control over evaluation logic:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset

# Create dataset
dataset = create_qa_dataset(
    questions=["What is Python?", "What is JavaScript?"],
    answers=["A programming language", "A programming language"]
)

@evaluate("gpt-4")
async def custom_evaluation(model, dataset):
    responses = []

    for prompt in dataset.prompts:
        # Custom prompt engineering
        enhanced_prompt = f"Answer concisely: {prompt}"

        # Generate with specific params
        response = await model.generate([enhanced_prompt], temperature=0.5)
        responses.extend(response)

    # Custom scoring logic
    scores = []
    for response, reference in zip(responses, dataset.references):
        # Your custom scoring
        score = len(response) > 0  # Simple example
        scores.append(score)

    return {
        "custom_score": sum(scores) / len(scores),
        "total": len(scores)
    }

# Run the evaluation
results = asyncio.run(custom_evaluation(dataset))

# Print results
for result in results:
    if result.success:
        print(f"{result.model_name}:")
        print(f"  Custom Score: {result.result['custom_score']:.2%}")
        print(f"  Total: {result.result['total']}")
    else:
        print(f"{result.model_name}: FAILED - {result.error}")
```

## Batch Processing

Handle large datasets efficiently:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a larger dataset
questions = [f"What is {i}+{i}?" for i in range(1, 21)]
answers = [str(i+i) for i in range(1, 21)]
dataset = create_qa_dataset(questions=questions, answers=answers)

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

# Run the evaluation
results = asyncio.run(batch_evaluation(dataset))

# Print results
for result in results:
    if result.success:
        print(f"{result.model_name}: Accuracy = {result.result['accuracy']:.2%}")
    else:
        print(f"{result.model_name}: FAILED - {result.error}")
```

## Result Upload

Enable automatic result upload (when platform is available):

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset
dataset = create_qa_dataset(
    questions=["What is machine learning?"],
    answers=["A subset of AI that learns from data"]
)

@evaluate("gpt-4", upload=True)
async def test_with_upload(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run the evaluation (results will be uploaded if configured)
results = asyncio.run(test_with_upload(dataset))

# Print results
for result in results:
    if result.success:
        print(f"{result.model_name}: Accuracy = {result.result['accuracy']:.2%}")
        print("Results uploaded to platform")
    else:
        print(f"{result.model_name}: FAILED - {result.error}")
```

## Saving Results

Save evaluation results for later analysis:

```python
import asyncio
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy, save_results, BenchmarkResult

# Create dataset
dataset = create_qa_dataset(
    questions=["What is aspirin used for?", "What is the normal body temperature?"],
    answers=["Pain relief and reducing fever", "98.6°F or 37°C"]
)

@benchmark("Medical QA v1.0", "Medical question answering evaluation")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }

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

    print("Results saved successfully!")
    print("- results.json (JSON format)")
    print("- results.csv (CSV format)")
    print("- report.md (Markdown report)")

asyncio.run(run_and_save())
```

## Best Practices

### 1. Use Descriptive Names

```python
import asyncio
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy

# Create customer support dataset
dataset = create_qa_dataset(
    questions=["How do I reset my password?", "What are your business hours?"],
    answers=["Click 'Forgot Password' on login page", "9 AM to 5 PM Monday-Friday"]
)

@benchmark("customer_support_qa_v2", "Customer support QA - Updated 2024")
@evaluate("gpt-4")
async def test_customer_support(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run the benchmark
results = asyncio.run(test_customer_support(dataset))

# Print results
for result in results:
    if result.success:
        print(f"{result.model_name}: Accuracy = {result.result['accuracy']:.2%}")
```

### 2. Set Temperature Appropriately

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Factual dataset
factual_dataset = create_qa_dataset(
    questions=["What is the capital of Germany?"],
    answers=["Berlin"]
)

# Creative dataset
creative_dataset = create_qa_dataset(
    questions=["Write a poem about the ocean"],
    answers=["A creative poem about the ocean"]
)

# For factual tasks - use temperature=0
@evaluate("gpt-4", temperature=0)
async def test_facts(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# For creative tasks - use higher temperature
@evaluate("gpt-4", temperature=0.9)
async def test_creative(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"response": responses[0]}

# Run both tests
factual_results = asyncio.run(test_facts(factual_dataset))
creative_results = asyncio.run(test_creative(creative_dataset))

print("Factual test (temperature=0):")
for result in factual_results:
    if result.success:
        print(f"  {result.model_name}: {result.result['accuracy']:.2%}")

print("\nCreative test (temperature=0.9):")
for result in creative_results:
    if result.success:
        print(f"  {result.model_name}: {result.result['response'][:80]}...")
```

### 3. Return Comprehensive Metrics

```python
import asyncio
import time
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset
dataset = create_qa_dataset(
    questions=["What is 5+5?", "What is 10-3?", "What is 2*6?"],
    answers=["10", "7", "12"]
)

@evaluate("gpt-4")
async def comprehensive_test(model, dataset):
    start_time = time.time()
    responses = await model.generate(dataset.prompts)
    duration = time.time() - start_time

    scores = accuracy(responses, dataset.references)

    return {
        "accuracy": scores["accuracy"],
        "total_samples": len(responses),
        "avg_length": sum(len(r) for r in responses) / len(responses),
        "duration": duration
    }

# Run the evaluation
results = asyncio.run(comprehensive_test(dataset))

# Print comprehensive metrics
for result in results:
    if result.success:
        print(f"{result.model_name} Results:")
        print(f"  Accuracy: {result.result['accuracy']:.2%}")
        print(f"  Total Samples: {result.result['total_samples']}")
        print(f"  Avg Response Length: {result.result['avg_length']:.1f} chars")
        print(f"  Duration: {result.result['duration']:.2f}s")
```

### 4. Test with Samples First

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a large dataset
questions = [f"What is {i}*2?" for i in range(1, 101)]
answers = [str(i*2) for i in range(1, 101)]
full_dataset = create_qa_dataset(questions=questions, answers=answers)

@evaluate("gpt-3.5-turbo")
async def my_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"], "total": len(responses)}

# Test with small sample first
print("Testing with sample...")
test_sample = full_dataset.sample(n=10, random_state=42)
results = asyncio.run(my_test(test_sample))

for result in results:
    if result.success:
        print(f"{result.model_name}: {result.result['accuracy']:.2%} on {result.result['total']} samples")

# Then run on full dataset
print("\nTesting with full dataset...")
results = asyncio.run(my_test(full_dataset))

for result in results:
    if result.success:
        print(f"{result.model_name}: {result.result['accuracy']:.2%} on {result.result['total']} samples")
```

## Next Steps

- [Metrics Guide](./metrics.md) - Learn about evaluation metrics
- [Datasets Guide](./datasets.md) - Master dataset management
- [Models Guide](./models.md) - Understand model adapters

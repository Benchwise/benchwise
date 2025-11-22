---
sidebar_position: 2
---

# Benchmark

Decorator to mark evaluation functions as named benchmarks with metadata.

## Signature

```python
@benchmark(name, description="", **kwargs)
@evaluate(*models)
async def evaluate_function(model, dataset):
    ...
```

## Parameters

- **`name`** (str): Name of the benchmark
- **`description`** (str, optional): Description of what the benchmark tests
- **`**kwargs`**: Additional keyword arguments passed to the benchmark.

## Returns

A decorator that adds benchmark metadata to the evaluation function.

## Basic Usage

```python
from benchwise import benchmark, evaluate

@benchmark("General QA", "Tests general knowledge questions")
@evaluate("gpt-4", "claude-3-opus")
async def test_general_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## With Metadata

```python
@benchmark(
    "Medical QA v2.0",
    "Medical question answering benchmark",
    version="2.0",
    domain="healthcare",
    difficulty="hard",
    dataset_size=100
)
@evaluate("gpt-4")
async def test_medical_qa(model, dataset):
    ...
```

## Accessing Metadata

Benchmark metadata is stored in the function's `_benchmark_metadata` attribute:

```python
@benchmark("Test Benchmark", "Description", version="1.0")
@evaluate("gpt-4")
async def my_test(model, dataset):
    ...

# Access metadata
metadata = my_test._benchmark_metadata
print(metadata["name"])         # "Test Benchmark"
print(metadata["description"])  # "Description"
print(metadata["version"])      # "1.0"
```

## Decorator Order

`@benchmark` must be applied before `@evaluate`:

```python
# Correct order
@benchmark("My Benchmark", "Description")
@evaluate("gpt-4")
async def correct_test(model, dataset):
    ...

# Wrong order - will not work properly
@evaluate("gpt-4")
@benchmark("My Benchmark", "Description")
async def wrong_test(model, dataset):
    ...
```

## Complete Example

```python
import asyncio
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy

dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"]
)

@benchmark(
    name="AI Knowledge Test v1.0",
    description="Tests understanding of AI and ML concepts",
    version="1.0",
    category="technology",
    difficulty="beginner",
    language="english"
)
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def test_ai_knowledge(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)

    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }

# Run the benchmark
results = asyncio.run(test_ai_knowledge(dataset))

# Access benchmark metadata
print(f"Benchmark: {test_ai_knowledge._benchmark_metadata['name']}")
print(f"Version: {test_ai_knowledge._benchmark_metadata['version']}")

# Process results
for result in results:
    if result.success:
        print(f"{result.model_name}: {result.result['accuracy']:.2%}")
```

## Versioning Benchmarks

Track benchmark versions over time:

```python
@benchmark("Customer QA", "Customer support QA", version="1.0")
@evaluate("gpt-3.5-turbo")
async def test_customer_qa_v1(model, dataset):
    # Original version
    pass

@benchmark("Customer QA", "Customer support QA", version="2.0")
@evaluate("gpt-4")
async def test_customer_qa_v2(model, dataset):
    # Updated version with improvements
    pass
```

## Domain-Specific Metadata

```python
# Medical domain
@benchmark(
    "Medical Diagnosis",
    "Diagnostic accuracy evaluation",
    domain="healthcare",
    specialty="general_medicine",
    risk_level="high"
)
@evaluate("gpt-4")
async def test_medical(model, dataset):
    ...

# Legal domain
@benchmark(
    "Legal Analysis",
    "Contract analysis benchmark",
    domain="legal",
    jurisdiction="US",
    contract_type="commercial"
)
@evaluate("gpt-4")
async def test_legal(model, dataset):
    ...

# Financial domain
@benchmark(
    "Financial Forecasting",
    "Stock price prediction",
    domain="finance",
    market="NYSE",
    timeframe="daily"
)
@evaluate("gpt-4")
async def test_financial(model, dataset):
    pass
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
@benchmark("Medical QA - Cardiology", "Heart disease diagnosis questions")
@evaluate("gpt-4")
async def test_medical_cardiology(model, dataset):
    ...

# Less descriptive
@benchmark("Test 1", "Some test")
@evaluate("gpt-4")
async def test_one(model, dataset):
    ...
```

### 2. Include Version Information

```python
@benchmark("Product QA", "Product questions", version="2.1", updated="2024-11-16")
@evaluate("gpt-4")
async def test_product_qa(model, dataset):
    ...
```

### 3. Document Difficulty

```python
@benchmark("Math Problems", "Algebra questions", difficulty="intermediate", grade_level="9-10")
@evaluate("gpt-4")
async def test_math_problems(model, dataset):
    ...
```

### 4. Specify Dataset Information

```python
@benchmark(
    "MMLU Sample",
    "Multiple choice questions",
    dataset_size=100,
    source="MMLU",
    sample_strategy="random"
)
@evaluate("gpt-4")
async def test_mmlu(model, dataset):
    ...
```

## See Also

- [@evaluate](./evaluate.md) - Main evaluation decorator
- [@stress_test](./stress-test.md) - Performance testing
- [Evaluation Guide](../../guides/evaluation.md) - Evaluation patterns

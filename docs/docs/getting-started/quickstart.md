---
sidebar_position: 2
---

# Quickstart

Run your first LLM evaluation in under 5 minutes.

## Your First Evaluation

Create a simple evaluation to compare models on basic questions:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a simple dataset
dataset = create_qa_dataset(
    questions=["What is the capital of France?", "What is 2+2?"],
    answers=["Paris", "4"]
)

# Evaluate multiple models
@evaluate("gpt-4o-mini", "claude-3-5-haiku-20241022")
async def test_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run it
results = asyncio.run(test_qa(dataset))
for result in results:
    print(f"{result.model_name}: {result.result['accuracy']:.2%}")
```

## What Just Happened?

Let's break down what this code does:

1. **Created a Dataset** - `create_qa_dataset()` creates a simple question-answer dataset
2. **Decorated a Function** - `@evaluate()` runs your test on multiple models
3. **Generated Responses** - `model.generate()` calls the LLM APIs
4. **Evaluated Results** - `accuracy()` compares responses to expected answers
5. **Got Results** - Each model returns an `EvaluationResult` with metrics

## Add a Benchmark

Make your evaluation reusable by marking it as a benchmark:

```python
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy

dataset = create_qa_dataset(
    questions=["What is AI?", "Explain machine learning"],
    answers=["Artificial Intelligence", "ML is a subset of AI that learns from data"]
)

@benchmark("AI Knowledge", "Tests basic AI understanding")
@evaluate("gpt-4", "claude-opus-4-1")
async def test_ai_knowledge(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

results = asyncio.run(test_ai_knowledge(dataset))
```

## Use Multiple Metrics

Evaluate with more than just accuracy:

```python
from benchwise import evaluate, create_qa_dataset, accuracy, semantic_similarity

dataset = create_qa_dataset(
    questions=["Explain photosynthesis"],
    answers=["Plants convert sunlight into energy through photosynthesis"]
)

@evaluate("gpt-4o-mini", "claude-3-5-haiku-20241022")
async def test_with_metrics(model, dataset):
    responses = await model.generate(dataset.prompts)

    # Multiple metrics
    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "semantic_similarity": similarity["mean_similarity"]
    }

results = asyncio.run(test_with_metrics(dataset))
```

## Next Steps

- [Core Concepts](./core-concepts.md) - Learn about decorators, datasets, and metrics
- [Evaluation Guide](../guides/evaluation.md) - Deep dive into evaluations
- [Examples](../examples/question-answering.md) - See real-world examples

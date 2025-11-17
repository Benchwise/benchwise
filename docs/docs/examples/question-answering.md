---
sidebar_position: 1
---

# Question Answering

Evaluate models on question answering tasks.

## Basic QA Evaluation

```python
import asyncio
from benchwise import evaluate, benchmark, create_qa_dataset, accuracy, semantic_similarity

# Create QA dataset
qa_dataset = create_qa_dataset(
    questions=[
        "What is the capital of Japan?",
        "Who wrote '1984'?",
        "What is the speed of light?",
        "Explain photosynthesis in one sentence.",
        "What causes rainbows?"
    ],
    answers=[
        "Tokyo",
        "George Orwell",
        "299,792,458 meters per second",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Rainbows are caused by light refraction and reflection in water droplets."
    ],
    name="general_knowledge_qa"
)

@benchmark("General Knowledge QA", "Tests basic factual knowledge")
@evaluate("gpt-3.5-turbo", "claude-3-5-haiku-20241022", "gemini-pro")
async def test_general_knowledge(model, dataset):
    responses = await model.generate(dataset.prompts)

    # Multiple metrics for comprehensive evaluation
    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "semantic_similarity": similarity["mean_similarity"],
        "total_questions": len(responses)
    }

# Run the evaluation
async def main():
    results = await test_general_knowledge(qa_dataset)

    print("\n=== General Knowledge QA Results ===")
    for result in results:
        if result.success:
            print(f"{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Similarity: {result.result['semantic_similarity']:.3f}")
            print(f"  Duration: {result.duration:.2f}s")
        else:
            print(f"{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Medical QA Example

```python
from benchwise import evaluate, benchmark, create_qa_dataset, accuracy, semantic_similarity

# Medical domain dataset
medical_qa = create_qa_dataset(
    questions=[
        "What is hypertension?",
        "What are the symptoms of diabetes?",
        "How does aspirin work?",
        "What is the function of the liver?"
    ],
    answers=[
        "High blood pressure",
        "Increased thirst, frequent urination, fatigue, and blurred vision",
        "Aspirin inhibits enzymes that produce prostaglandins, reducing inflammation and pain",
        "The liver filters blood, produces bile, metabolizes nutrients, and detoxifies harmful substances"
    ],
    name="medical_qa"
)

@benchmark("Medical QA", "Medical question answering benchmark")
@evaluate("gpt-4", "claude-3-opus")
async def test_medical_qa(model, dataset):
    # Use temperature=0 for factual accuracy
    responses = await model.generate(dataset.prompts, temperature=0)

    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "similarity": similarity["mean_similarity"]
    }

asyncio.run(test_medical_qa(medical_qa))
```

## Multi-Hop Reasoning

```python
from benchwise import evaluate, create_qa_dataset, accuracy

# Complex reasoning questions
reasoning_qa = create_qa_dataset(
    questions=[
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A train leaves New York at 2 PM traveling at 60 mph. Another leaves Boston at 3 PM traveling at 80 mph toward New York. If they're 200 miles apart, when will they meet?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?"
    ],
    answers=[
        "No, this is a logical fallacy. We cannot conclude that some roses fade quickly.",
        "They will meet at 4:30 PM",
        "5 minutes"
    ],
    name="reasoning_qa"
)

@evaluate("gpt-4", "claude-3-opus", temperature=0)
async def test_reasoning(model, dataset):
    # Add reasoning instruction
    prompts = [f"Think step by step and answer: {q}" for q in dataset.prompts]
    responses = await model.generate(prompts)

    acc = accuracy(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "total": len(responses)
    }

asyncio.run(test_reasoning(reasoning_qa))
```

## Handling Ambiguous Questions

```python
from benchwise import evaluate, create_qa_dataset

ambiguous_qa = create_qa_dataset(
    questions=[
        "What is the capital of Congo?",  # Two countries named Congo
        "Who invented the telephone?",     # Disputed invention
        "What is the tallest mountain?"    # On Earth? In solar system?
    ],
    answers=[
        "Kinshasa (DRC) or Brazzaville (Republic of Congo)",
        "Alexander Graham Bell (though disputed with Antonio Meucci)",
        "Mount Everest (on Earth)"
    ],
    name="ambiguous_qa"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_ambiguous(model, dataset):
    # Ask models to clarify ambiguity
    prompts = [f"Answer this question and note any ambiguities: {q}"
               for q in dataset.prompts]
    responses = await model.generate(prompts)

    # Custom scoring: check if model acknowledges ambiguity
    ambiguity_scores = []
    for response in responses:
        response_lower = response.lower()
        acknowledges = any(word in response_lower
                          for word in ["ambiguous", "unclear", "depends", "could be", "both"])
        ambiguity_scores.append(1 if acknowledges else 0)

    return {
        "ambiguity_acknowledgment_rate": sum(ambiguity_scores) / len(ambiguity_scores),
        "total": len(responses)
    }

asyncio.run(test_ambiguous(ambiguous_qa))
```

## Batch Processing Large QA Datasets

```python
from benchwise import evaluate, load_dataset, accuracy

# Load large dataset
large_qa = load_dataset("data/qa_1000.json")

@evaluate("gpt-3.5-turbo")
async def test_large_batch(model, dataset):
    # Sample for testing
    test_sample = dataset.sample(n=100, random_state=42)

    # Process in batches of 10
    batch_size = 10
    all_responses = []

    for i in range(0, len(test_sample.prompts), batch_size):
        batch = test_sample.prompts[i:i+batch_size]
        responses = await model.generate(batch)
        all_responses.extend(responses)

        print(f"Processed {len(all_responses)}/{len(test_sample.prompts)}")

    # Evaluate
    acc = accuracy(all_responses, test_sample.references)

    return {
        "accuracy": acc["accuracy"],
        "total_processed": len(all_responses)
    }

asyncio.run(test_large_batch(large_qa))
```

## Using Standard Benchmarks

```python
from benchwise import evaluate, load_mmlu_sample, accuracy

# Load MMLU sample
mmlu = load_mmlu_sample()

@benchmark("MMLU Sample", "Multiple choice questions from MMLU")
@evaluate("gpt-4", "claude-3-opus")
async def test_mmlu(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    acc = accuracy(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "total": len(responses)
    }

asyncio.run(test_mmlu(mmlu))
```

## Saving and Analyzing Results

```python
from benchwise import save_results, BenchmarkResult, ResultsAnalyzer

async def run_complete_qa_evaluation():
    results = await test_general_knowledge(qa_dataset)

    # Create benchmark result
    benchmark = BenchmarkResult(
        "Complete QA Evaluation",
        metadata={"date": "2024-11-16", "version": "1.0"}
    )

    for result in results:
        benchmark.add_result(result)

    # Save in multiple formats
    save_results(benchmark, "qa_results.json", format="json")
    save_results(benchmark, "qa_results.csv", format="csv")
    save_results(benchmark, "qa_report.md", format="markdown")

    # Generate analysis report
    report = ResultsAnalyzer.generate_report(benchmark, "markdown")
    print(report)

    # Compare models
    comparison = benchmark.compare_models("accuracy")
    print(f"\nBest model: {comparison['best_model']}")
    print(f"Best accuracy: {comparison['best_score']:.2%}")

asyncio.run(run_complete_qa_evaluation())
```

## Next Steps

- [Summarization Example](./summarization.md) - Text summarization evaluation
- [Multi-Model Comparison](./multi-model-comparison.md) - Compare multiple models
- [Metrics Guide](../guides/metrics.md) - Learn about evaluation metrics

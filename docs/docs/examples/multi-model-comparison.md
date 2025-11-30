---
sidebar_position: 5
---

# Multi-Model Comparison

Compare multiple models across different tasks and metrics.

## Comprehensive Model Comparison

```python
import asyncio
from benchwise import (
    evaluate,
    benchmark,
    create_qa_dataset,
    accuracy,
    semantic_similarity,
    save_results,
    BenchmarkResult,
    ResultsAnalyzer
)

# Create comprehensive dataset
qa_dataset = create_qa_dataset(
    questions=[
        "What is artificial intelligence?",
        "Explain quantum computing",
        "What causes climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ],
    answers=[
        "AI is the simulation of human intelligence in machines",
        "Quantum computing uses quantum-mechanical phenomena to perform calculations",
        "Climate change is caused by greenhouse gas emissions from human activities",
        "Photosynthesis is how plants convert light into chemical energy",
        "Einstein's theory describing the relationship between space, time, and gravity"
    ],
    name="comprehensive_qa"
)

@benchmark("Multi-Model QA Comparison", "Compare models across accuracy and similarity")
@evaluate(
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "gemini-pro"
)
async def compare_all_models(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)

    # Multiple metrics
    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "semantic_similarity": similarity["mean_similarity"],
        "total_questions": len(responses)
    }

async def main():
    print("Running comprehensive model comparison...")
    results = await compare_all_models(qa_dataset)

    # Create benchmark result
    benchmark = BenchmarkResult(
        "Multi-Model Comparison",
        metadata={
            "date": "2024-11-16",
            "task": "qa",
            "models_tested": 6
        }
    )

    for result in results:
        benchmark.add_result(result)

    # Print results
    print("\n=== Model Comparison Results ===\n")

    for result in sorted(results, key=lambda r: r.result.get("accuracy", 0), reverse=True):
        if result.success:
            print(f"{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Similarity: {result.result['semantic_similarity']:.3f}")
            print(f"  Duration: {result.duration:.2f}s")
            print()

    # Find best performers
    accuracy_comparison = benchmark.compare_models("accuracy")
    similarity_comparison = benchmark.compare_models("semantic_similarity")

    print("=== Best Performers ===")
    print(f"Best Accuracy: {accuracy_comparison['best_model']} ({accuracy_comparison['best_score']:.2%})")
    print(f"Best Similarity: {similarity_comparison['best_model']} ({similarity_comparison['best_score']:.3f})")

    # Save results
    save_results(benchmark, "multi_model_results.json", format="json")
    save_results(benchmark, "multi_model_results.csv", format="csv")
    save_results(benchmark, "multi_model_report.md", format="markdown")

    # Generate detailed report
    report = ResultsAnalyzer.generate_report(benchmark, "markdown")
    with open("detailed_report.md", "w") as f:
        f.write(report)

    print("\nResults saved to files.")

asyncio.run(main())
```

## Cost vs Performance Analysis

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset for testing
qa_dataset = create_qa_dataset(
    questions=[
        "What is artificial intelligence?",
        "Explain quantum computing",
        "What causes climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ],
    answers=[
        "AI is the simulation of human intelligence in machines",
        "Quantum computing uses quantum-mechanical phenomena to perform calculations",
        "Climate change is caused by greenhouse gas emissions from human activities",
        "Photosynthesis is how plants convert light into chemical energy",
        "Einstein's theory describing the relationship between space, time, and gravity"
    ],
    name="comprehensive_qa"
)

@evaluate("gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-haiku")
async def compare_cost_performance(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)

    # Calculate cost
    input_tokens = sum(model.get_token_count(p) for p in dataset.prompts)
    output_tokens = sum(model.get_token_count(r) for r in responses)
    estimated_cost = model.get_cost_estimate(input_tokens, output_tokens)

    # Calculate performance
    acc = accuracy(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "cost": estimated_cost,
        "cost_per_query": estimated_cost / len(dataset.prompts),
        "cost_per_accuracy_point": estimated_cost / acc["accuracy"] if acc["accuracy"] > 0 else float('inf')
    }

async def analyze_cost_performance():
    results = await compare_cost_performance(qa_dataset)

    print("\n=== Cost vs Performance ===\n")

    for result in results:
        if result.success:
            print(f"{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Total Cost: ${result.result['cost']:.4f}")
            print(f"  Cost per Query: ${result.result['cost_per_query']:.4f}")
            print(f"  Cost per Accuracy Point: ${result.result['cost_per_accuracy_point']:.4f}")
            print()

    # Find best value
    best_value = min(
        [r for r in results if r.success],
        key=lambda r: r.result['cost_per_accuracy_point']
    )

    print(f"Best Value: {best_value.model_name}")

asyncio.run(analyze_cost_performance())
```

## Speed vs Accuracy Trade-offs

```python
import asyncio
import time
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset for testing
qa_dataset = create_qa_dataset(
    questions=[
        "What is artificial intelligence?",
        "Explain quantum computing",
        "What causes climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ],
    answers=[
        "AI is the simulation of human intelligence in machines",
        "Quantum computing uses quantum-mechanical phenomena to perform calculations",
        "Climate change is caused by greenhouse gas emissions from human activities",
        "Photosynthesis is how plants convert light into chemical energy",
        "Einstein's theory describing the relationship between space, time, and gravity"
    ],
    name="comprehensive_qa"
)

@evaluate("gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-haiku", "gemini-pro")
async def compare_speed_accuracy(model, dataset):
    start_time = time.time()
    responses = await model.generate(dataset.prompts, temperature=0)
    duration = time.time() - start_time

    acc = accuracy(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "duration": duration,
        "queries_per_second": len(dataset.prompts) / duration,
        "seconds_per_query": duration / len(dataset.prompts)
    }

async def analyze_speed_accuracy():
    results = await compare_speed_accuracy(qa_dataset)

    print("\n=== Speed vs Accuracy ===\n")

    # Sort by queries per second
    for result in sorted(results, key=lambda r: r.result.get("queries_per_second", 0), reverse=True):
        if result.success:
            print(f"{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Duration: {result.result['duration']:.2f}s")
            print(f"  Queries/sec: {result.result['queries_per_second']:.2f}")
            print(f"  Sec/query: {result.result['seconds_per_query']:.3f}")
            print()

    # Find balanced model
    # Score = accuracy / (seconds_per_query * 10) - higher is better
    for result in results:
        if result.success:
            result.balance_score = result.result['accuracy'] / (result.result['seconds_per_query'] * 10)

    best_balanced = max(
        [r for r in results if r.success],
        key=lambda r: r.balance_score
    )

    print(f"Best Balanced: {best_balanced.model_name}")

asyncio.run(analyze_speed_accuracy())
```

## Task-Specific Comparisons

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, create_summarization_dataset, accuracy, rouge_l

# Different tasks
qa_data = create_qa_dataset(
    questions=["What is AI?", "Explain ML"],
    answers=["Artificial Intelligence", "Machine Learning"],
    name="qa_task"
)

summ_data = create_summarization_dataset(
    documents=["Long article about AI...", "Long article about ML..."],
    summaries=["AI summary", "ML summary"],
    name="summarization_task"
)

@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def compare_on_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    acc = accuracy(responses, dataset.references)
    return {"accuracy": acc["accuracy"]}

@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def compare_on_summarization(model, dataset):
    prompts = [f"Summarize: {doc}" for doc in dataset.prompts]
    responses = await model.generate(prompts)
    rouge = rouge_l(responses, dataset.references)
    return {"rouge_f1": rouge["f1"]}

async def task_specific_comparison():
    print("Testing on QA task...")
    qa_results = await compare_on_qa(qa_data)

    print("Testing on summarization task...")
    summ_results = await compare_on_summarization(summ_data)

    print("\n=== Task-Specific Results ===\n")

    print("QA Task:")
    for result in qa_results:
        if result.success:
            print(f"  {result.model_name}: {result.result['accuracy']:.2%}")

    print("\nSummarization Task:")
    for result in summ_results:
        if result.success:
            print(f"  {result.model_name}: {result.result['rouge_f1']:.3f}")

asyncio.run(task_specific_comparison())
```

## Provider Comparison

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset for testing
qa_dataset = create_qa_dataset(
    questions=[
        "What is artificial intelligence?",
        "Explain quantum computing",
        "What causes climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ],
    answers=[
        "AI is the simulation of human intelligence in machines",
        "Quantum computing uses quantum-mechanical phenomena to perform calculations",
        "Climate change is caused by greenhouse gas emissions from human activities",
        "Photosynthesis is how plants convert light into chemical energy",
        "Einstein's theory describing the relationship between space, time, and gravity"
    ],
    name="comprehensive_qa"
)

# Compare providers
openai_models = ["gpt-4", "gpt-3.5-turbo"]
anthropic_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
google_models = ["gemini-pro"]

@evaluate(*openai_models, *anthropic_models, *google_models)
async def compare_providers(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    acc = accuracy(responses, dataset.references)

    return {"accuracy": acc["accuracy"]}

async def provider_analysis():
    results = await compare_providers(qa_dataset)

    # Group by provider
    providers = {
        "OpenAI": [],
        "Anthropic": [],
        "Google": []
    }

    for result in results:
        if result.success:
            if "gpt" in result.model_name:
                providers["OpenAI"].append(result)
            elif "claude" in result.model_name:
                providers["Anthropic"].append(result)
            elif "gemini" in result.model_name:
                providers["Google"].append(result)

    print("\n=== Provider Comparison ===\n")

    for provider, provider_results in providers.items():
        if provider_results:
            avg_accuracy = sum(r.result["accuracy"] for r in provider_results) / len(provider_results)
            print(f"{provider}:")
            print(f"  Average Accuracy: {avg_accuracy:.2%}")
            print(f"  Models tested: {len(provider_results)}")
            for r in provider_results:
                print(f"    - {r.model_name}: {r.result['accuracy']:.2%}")
            print()

asyncio.run(provider_analysis())
```

## Model Size Comparison

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset for testing
qa_dataset = create_qa_dataset(
    questions=[
        "What is artificial intelligence?",
        "Explain quantum computing",
        "What causes climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ],
    answers=[
        "AI is the simulation of human intelligence in machines",
        "Quantum computing uses quantum-mechanical phenomena to perform calculations",
        "Climate change is caused by greenhouse gas emissions from human activities",
        "Photosynthesis is how plants convert light into chemical energy",
        "Einstein's theory describing the relationship between space, time, and gravity"
    ],
    name="comprehensive_qa"
)

# Compare model sizes within same provider
small_models = ["gpt-3.5-turbo", "claude-3-haiku"]
large_models = ["gpt-4", "claude-3-opus"]

@evaluate(*small_models, *large_models)
async def compare_model_sizes(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)

    # Estimate tokens
    input_tokens = sum(model.get_token_count(p) for p in dataset.prompts)
    output_tokens = sum(model.get_token_count(r) for r in responses)

    acc = accuracy(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }

async def size_analysis():
    results = await compare_model_sizes(qa_dataset)

    print("\n=== Model Size Comparison ===\n")

    print("Small Models:")
    for result in results:
        if result.success and result.model_name in small_models:
            print(f"  {result.model_name}:")
            print(f"    Accuracy: {result.result['accuracy']:.2%}")
            print(f"    Total Tokens: {result.result['total_tokens']}")

    print("\nLarge Models:")
    for result in results:
        if result.success and result.model_name in large_models:
            print(f"  {result.model_name}:")
            print(f"    Accuracy: {result.result['accuracy']:.2%}")
            print(f"    Total Tokens: {result.result['total_tokens']}")

asyncio.run(size_analysis())
```

## Comprehensive Leaderboard

```python
import asyncio
from benchwise import evaluate, benchmark, create_qa_dataset, accuracy, semantic_similarity, save_results, BenchmarkResult

# Create dataset for testing
qa_dataset = create_qa_dataset(
    questions=[
        "What is artificial intelligence?",
        "Explain quantum computing",
        "What causes climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ],
    answers=[
        "AI is the simulation of human intelligence in machines",
        "Quantum computing uses quantum-mechanical phenomena to perform calculations",
        "Climate change is caused by greenhouse gas emissions from human activities",
        "Photosynthesis is how plants convert light into chemical energy",
        "Einstein's theory describing the relationship between space, time, and gravity"
    ],
    name="comprehensive_qa"
)

@benchmark("Model Leaderboard", "Comprehensive model ranking")
@evaluate("gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "gemini-pro")
async def create_leaderboard(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)

    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    # Composite score: weighted average
    composite_score = (acc["accuracy"] * 0.6) + (similarity["mean_similarity"] * 0.4)

    return {
        "accuracy": acc["accuracy"],
        "similarity": similarity["mean_similarity"],
        "composite_score": composite_score,
        "rank": 0  # Will be filled later
    }

async def generate_leaderboard():
    results = await create_leaderboard(qa_dataset)

    # Sort by composite score
    successful_results = [r for r in results if r.success]
    successful_results.sort(key=lambda r: r.result["composite_score"], reverse=True)

    # Assign ranks
    for rank, result in enumerate(successful_results, 1):
        result.result["rank"] = rank

    print("\n=== MODEL LEADERBOARD ===\n")
    print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<12} {'Similarity':<12} {'Score':<10}")
    print("-" * 80)

    for result in successful_results:
        print(f"{result.result['rank']:<6} "
              f"{result.model_name:<30} "
              f"{result.result['accuracy']:<12.2%} "
              f"{result.result['similarity']:<12.3f} "
              f"{result.result['composite_score']:<10.3f}")

    # Save leaderboard
    benchmark = BenchmarkResult("Model Leaderboard")
    for result in successful_results:
        benchmark.add_result(result)

    save_results(benchmark, "leaderboard.json", format="json")
    save_results(benchmark, "leaderboard.md", format="markdown")

asyncio.run(generate_leaderboard())
```

## Next Steps

- [Question Answering](./question-answering.md) - QA-specific examples
- [Guides](../guides/evaluation.md) - Learn evaluation best practices
- [Results Guide](../guides/results.md) - Analyze and compare results

---
sidebar_position: 9
---

# Results

Complete, runnable examples demonstrating result management, analysis, and export in Benchwise.

Each example below is self-contained and can be copied and run directly.

## Scenario 1: Basic Result Handling

Understanding and accessing evaluation results.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset
dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"]
)

@evaluate("gpt-3.5-turbo", "gpt-4")
async def basic_eval(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run evaluation
results = asyncio.run(basic_eval(dataset))

# Access result properties
print("=== Evaluation Results ===\n")
for result in results:
    print(f"Model: {result.model_name}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration:.2f}s")

    if result.success:
        print(f"Accuracy: {result.result['accuracy']:.2%}")
    else:
        print(f"Error: {result.error}")

    if result.metadata:
        print(f"Metadata: {result.metadata}")

    print("-" * 50)
```

## Scenario 2: Checking for Failures

Handle evaluation failures gracefully.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

dataset = create_qa_dataset(
    questions=["What is Python?"],
    answers=["A programming language"]
)

@evaluate("gpt-4", "invalid-model-name", "claude-3-opus")
async def test_with_failures(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run evaluation
results = asyncio.run(test_with_failures(dataset))

# Separate successful and failed results
successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"Total evaluations: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}\n")

# Report failures
if failed:
    print("=== Failed Evaluations ===")
    for failure in failed:
        print(f"❌ {failure.model_name}: {failure.error}")

# Report successes
if successful:
    print("\n=== Successful Evaluations ===")
    for success in successful:
        print(f"✅ {success.model_name}: {success.result['accuracy']:.2%}")
```

## Scenario 3: Creating Benchmark Results

Organize multiple evaluations into a benchmark.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy, BenchmarkResult

dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?", "What is DL?"],
    answers=["Artificial Intelligence", "Machine Learning", "Deep Learning"]
)

@evaluate("gpt-3.5-turbo", "gpt-4", "claude-3-opus")
async def ai_knowledge_test(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run evaluation
results = asyncio.run(ai_knowledge_test(dataset))

# Create benchmark result container
benchmark = BenchmarkResult(
    benchmark_name="AI Knowledge Benchmark",
    metadata={
        "date": "2024-11-28",
        "version": "1.0",
        "dataset_size": len(dataset.data),
        "temperature": 0
    }
)

# Add results to benchmark
for result in results:
    benchmark.add_result(result)

print(f"Benchmark: {benchmark.benchmark_name}")
print(f"Total results: {len(benchmark.results)}")
print(f"Metadata: {benchmark.metadata}")

# Access results
print("\n=== Results ===")
for result in benchmark.results:
    if result.success:
        print(f"{result.model_name}: {result.result['accuracy']:.2%}")
```

## Scenario 4: Saving Results as JSON

Save results in JSON format for later analysis.

```python
import asyncio
from benchwise import evaluate, benchmark, create_qa_dataset, accuracy, BenchmarkResult, save_results

dataset = create_qa_dataset(
    questions=["What is Python?", "What is JavaScript?"],
    answers=["A programming language", "A programming language"]
)

@benchmark("Programming Languages QA", "Test knowledge of programming languages")
@evaluate("gpt-3.5-turbo", "gpt-4")
async def programming_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run and collect results
results = asyncio.run(programming_test(dataset))

# Create benchmark container
benchmark_obj = BenchmarkResult(benchmark_name="Programming QA Results")
for result in results:
    benchmark_obj.add_result(result)

# Save as JSON
save_results(benchmark_obj, "programming_results.json", format="json")
print("✅ Results saved to programming_results.json")

# Show what was saved
import json
with open("programming_results.json", 'r') as f:
    saved_data = json.load(f)
    print(f"\nSaved data structure:")
    print(f"  Name: {saved_data.get('benchmark_name')}")
    print(f"  Total results: {len(saved_data.get('results', []))}")

# Cleanup
import os
os.remove("programming_results.json")
print("\n✅ Cleanup complete")
```

## Scenario 5: Saving Results as CSV

Export results in CSV format for spreadsheet analysis.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy, BenchmarkResult, save_results

dataset = create_qa_dataset(
    questions=["Q1?", "Q2?", "Q3?"],
    answers=["A1", "A2", "A3"]
)

@evaluate("gpt-3.5-turbo", "gpt-4", "claude-3-opus")
async def csv_export_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }

# Run evaluation
results = asyncio.run(csv_export_test(dataset))

# Create benchmark
benchmark = BenchmarkResult(benchmark_name="CSV Export Example")
for result in results:
    benchmark.add_result(result)

# Save as CSV
save_results(benchmark, "results.csv", format="csv")
print("✅ Results saved to results.csv")

# Show CSV contents
with open("results.csv", 'r') as f:
    print("\nCSV Contents:")
    print(f.read())

# Cleanup
import os
os.remove("results.csv")
print("✅ Cleanup complete")
```

## Scenario 6: Generating Markdown Reports

Create formatted markdown reports from results.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy, BenchmarkResult, save_results

dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"]
)

@evaluate("gpt-3.5-turbo", "gpt-4")
async def markdown_report_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run evaluation
results = asyncio.run(markdown_report_test(dataset))

# Create benchmark with detailed metadata
benchmark = BenchmarkResult(
    benchmark_name="AI Knowledge Evaluation",
    metadata={
        "date": "2024-11-28",
        "evaluator": "BenchWise",
        "dataset_size": len(dataset.data)
    }
)

for result in results:
    benchmark.add_result(result)

# Save as markdown
save_results(benchmark, "report.md", format="markdown")
print("✅ Markdown report saved to report.md")

# Show report contents
with open("report.md", 'r') as f:
    print("\nMarkdown Report:\n")
    print(f.read())

# Cleanup
import os
os.remove("report.md")
print("\n✅ Cleanup complete")
```

## Scenario 7: Loading Saved Results

Load and reuse previously saved results.

```python
import asyncio
import json
from benchwise import evaluate, create_qa_dataset, accuracy, BenchmarkResult, save_results, load_results

# First, run an evaluation and save results
dataset = create_qa_dataset(
    questions=["What is AI?"],
    answers=["Artificial Intelligence"]
)

@evaluate("gpt-3.5-turbo")
async def initial_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

results = asyncio.run(initial_test(dataset))

# Save results
benchmark = BenchmarkResult(benchmark_name="Initial Test")
for result in results:
    benchmark.add_result(result)

save_results(benchmark, "saved_results.json", format="json")
print("✅ Results saved\n")

# Later... load the results
loaded_benchmark = load_results("saved_results.json")

print("=== Loaded Results ===")
print(f"Benchmark name: {loaded_benchmark.benchmark_name}")
print(f"Total results: {len(loaded_benchmark.results)}")

for result in loaded_benchmark.results:
    print(f"\n{result.model_name}:")
    print(f"  Success: {result.success}")
    print(f"  Accuracy: {result.result.get('accuracy', 'N/A')}")

# Cleanup
import os
os.remove("saved_results.json")
print("\n✅ Cleanup complete")
```

## Scenario 8: Comparing Models

Compare model performance across metrics.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy, BenchmarkResult

dataset = create_qa_dataset(
    questions=["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
    answers=["A1", "A2", "A3", "A4", "A5"]
)

@evaluate("gpt-3.5-turbo", "gpt-4", "claude-3-opus", "gemini-pro")
async def comparison_test(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "correct_count": scores["correct"]
    }

# Run evaluation
results = asyncio.run(comparison_test(dataset))

# Create benchmark
benchmark = BenchmarkResult(benchmark_name="Model Comparison")
for result in results:
    benchmark.add_result(result)

# Compare by accuracy
comparison = benchmark.compare_models("accuracy")

print("=== Model Comparison ===\n")
print(f"Best Model: {comparison['best_model']}")
print(f"Best Score: {comparison['best_score']:.2%}")
print(f"\nWorst Model: {comparison['worst_model']}")
print(f"Worst Score: {comparison['worst_score']:.2%}")

# Show all models ranked
print("\n=== All Models (Ranked by Accuracy) ===")
ranked = sorted(
    [(r.model_name, r.result['accuracy']) for r in benchmark.results if r.success],
    key=lambda x: x[1],
    reverse=True
)

for rank, (model, acc) in enumerate(ranked, 1):
    print(f"{rank}. {model}: {acc:.2%}")
```

## Scenario 9: Generating Analysis Reports

Use ResultsAnalyzer for comprehensive reporting.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy, semantic_similarity, BenchmarkResult, ResultsAnalyzer

dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?", "What is DL?"],
    answers=["Artificial Intelligence", "Machine Learning", "Deep Learning"]
)

@evaluate("gpt-3.5-turbo", "gpt-4", "claude-3-opus")
async def analysis_test(model, dataset):
    responses = await model.generate(dataset.prompts)

    acc = accuracy(responses, dataset.references)
    sim = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "similarity": sim["mean_similarity"]
    }

# Run evaluation
results = asyncio.run(analysis_test(dataset))

# Create benchmark
benchmark = BenchmarkResult(benchmark_name="Comprehensive Analysis")
for result in results:
    benchmark.add_result(result)

# Generate markdown report
print("=== Markdown Report ===\n")
markdown_report = ResultsAnalyzer.generate_report(benchmark, output_format="markdown")
print(markdown_report)

# Generate text report
print("\n=== Text Report ===\n")
text_report = ResultsAnalyzer.generate_report(benchmark, output_format="text")
print(text_report)

# Get statistics for accuracy
print("\n=== Accuracy Statistics ===")
stats = ResultsAnalyzer.get_statistics(benchmark, "accuracy")
print(f"Mean: {stats['mean']:.2%}")
print(f"Std Dev: {stats['std']:.3f}")
print(f"Min: {stats['min']:.2%}")
print(f"Max: {stats['max']:.2%}")
```

## Scenario 10: Result Caching

Work with Benchwise's automatic result caching.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy
from benchwise.results import cache

dataset = create_qa_dataset(
    questions=["What is caching?"],
    answers=["Temporary storage of data"]
)

@evaluate("gpt-3.5-turbo")
async def cached_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run evaluation (results will be cached)
print("Running evaluation (will be cached)...")
results_first = asyncio.run(cached_test(dataset))
print(f"First run: {results_first[0].result['accuracy']:.2%}")

# List cached results
cached_items = cache.list_cached_results()
print(f"\nCached evaluations: {len(cached_items)}")

# Run again (should use cache if available)
print("\nRunning same evaluation again...")
results_second = asyncio.run(cached_test(dataset))
print(f"Second run: {results_second[0].result['accuracy']:.2%}")

# Clear cache
cache.clear_cache()
print("\n✅ Cache cleared")

# Verify cache is empty
cached_after_clear = cache.list_cached_results()
print(f"Cached evaluations after clear: {len(cached_after_clear)}")
```

## Scenario 11: Complete Workflow

End-to-end example: evaluation, benchmarking, saving, and analysis.

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

# Step 1: Create dataset
dataset = create_qa_dataset(
    questions=[
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ],
    answers=[
        "A high-level programming language",
        "A scripting language for web development",
        "A systems programming language"
    ]
)

# Step 2: Define evaluation with benchmark metadata
@benchmark(
    name="Programming Languages Test v1.0",
    description="Evaluate model knowledge of programming languages"
)
@evaluate("gpt-3.5-turbo", "gpt-4", "claude-3-opus")
async def programming_languages_eval(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)

    # Calculate multiple metrics
    acc = accuracy(responses, dataset.references)
    sim = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "similarity": sim["mean_similarity"],
        "total_questions": len(responses)
    }

# Step 3: Run evaluation
async def main():
    print("Step 1: Running evaluation...")
    results = await programming_languages_eval(dataset)

    # Step 4: Create benchmark container
    print("\nStep 2: Creating benchmark container...")
    benchmark_obj = BenchmarkResult(
        benchmark_name="Programming Languages Benchmark",
        metadata={
            "date": "2024-11-28",
            "version": "1.0",
            "dataset_size": len(dataset.data),
            "models_tested": ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
        }
    )

    for result in results:
        benchmark_obj.add_result(result)

    # Step 5: Save in multiple formats
    print("\nStep 3: Saving results...")
    save_results(benchmark_obj, "final_results.json", format="json")
    save_results(benchmark_obj, "final_results.csv", format="csv")
    save_results(benchmark_obj, "final_report.md", format="markdown")
    print("✅ Saved: JSON, CSV, Markdown")

    # Step 6: Analyze results
    print("\nStep 4: Analyzing results...")
    comparison = benchmark_obj.compare_models("accuracy")
    print(f"\nBest performing model: {comparison['best_model']}")
    print(f"Best accuracy: {comparison['best_score']:.2%}")

    # Step 7: Generate comprehensive report
    print("\nStep 5: Generating report...\n")
    report = ResultsAnalyzer.generate_report(benchmark_obj, output_format="text")
    print(report)

    # Step 8: Get statistics
    print("\n=== Statistics ===")
    acc_stats = ResultsAnalyzer.get_statistics(benchmark_obj, "accuracy")
    print(f"Accuracy - Mean: {acc_stats['mean']:.2%}, Std: {acc_stats['std']:.3f}")

    sim_stats = ResultsAnalyzer.get_statistics(benchmark_obj, "similarity")
    print(f"Similarity - Mean: {sim_stats['mean']:.2%}, Std: {sim_stats['std']:.3f}")

    # Cleanup
    import os
    for file in ["final_results.json", "final_results.csv", "final_report.md"]:
        if os.path.exists(file):
            os.remove(file)
    print("\n✅ Cleanup complete")

asyncio.run(main())
```

## Best Practice: Comprehensive Result Tracking

Track results over time for continuous improvement.

```python
import asyncio
from datetime import datetime
from benchwise import evaluate, create_qa_dataset, accuracy, BenchmarkResult, save_results

dataset = create_qa_dataset(
    questions=["Q1?", "Q2?"],
    answers=["A1", "A2"]
)

@evaluate("gpt-4")
async def tracked_evaluation(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run evaluation
results = asyncio.run(tracked_evaluation(dataset))

# Create benchmark with comprehensive metadata
benchmark = BenchmarkResult(
    benchmark_name="Tracked Evaluation",
    metadata={
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        "dataset_size": len(dataset.data),
        "models": ["gpt-4"],
        "temperature": 0,
        "environment": "production",
        "git_commit": "abc123",  # Track code version
        "notes": "Baseline evaluation for new dataset"
    }
)

for result in results:
    benchmark.add_result(result)

# Save with timestamp in filename
filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
save_results(benchmark, filename, format="json")
print(f"✅ Results saved to {filename}")

# Show saved metadata
import json
with open(filename, 'r') as f:
    saved = json.load(f)
    print(f"\nSaved metadata: {saved.get('metadata')}")

# Cleanup
import os
os.remove(filename)
print("\n✅ Cleanup complete")
```

## Related Examples

- [Evaluation](./evaluation.md) - Complete evaluation workflows including result handling
- [Multi-Model Comparison](./multi-model-comparison.md) - Advanced model comparison techniques

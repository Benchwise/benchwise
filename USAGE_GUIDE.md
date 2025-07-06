# BenchWise SDK: Complete Usage Guide

Welcome to BenchWise - the GitHub of LLM evaluation! This guide will walk you through everything you need to know to get started with evaluating your language models.

## Quick Start

First, install BenchWise:

```bash
pip install benchwise
```

Here's your first evaluation in under 10 lines:

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a simple dataset
dataset = create_qa_dataset(
    questions=["What is the capital of France?", "What is 2+2?"],
    answers=["Paris", "4"]
)

@evaluate("gpt-3.5-turbo", "claude-3-5-haiku-20241022")
async def my_first_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run it
results = asyncio.run(my_first_test(dataset))
print(f"GPT-3.5: {results[0].result}")
print(f"Claude: {results[1].result}")
```

That's it! You just ran your first LLM evaluation.

## Core Concepts

### 1. The @evaluate Decorator

The `@evaluate` decorator is your main tool. It handles all the complexity of running evaluations across multiple models:

```python
# Single model
@evaluate("gpt-4")
async def test_reasoning(model, dataset):
    # Your test logic here
    pass

# Multiple models
@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def compare_models(model, dataset):
    # Test all three models
    pass

# With custom settings
@evaluate("gpt-4", upload=True, temperature=0.7)
async def creative_test(model, dataset):
    # Results will be uploaded to BenchWise API
    pass
```

### 2. Datasets

BenchWise makes it easy to work with evaluation datasets:

```python
from benchwise import create_qa_dataset, create_summarization_dataset, load_dataset

# Create datasets programmatically
qa_data = create_qa_dataset(
    questions=["What is AI?", "Explain machine learning"],
    answers=["Artificial Intelligence", "ML is a subset of AI..."]
)

# Load from files
dataset = load_dataset("my_dataset.json")  # or .csv

# Sample built-in datasets
from benchwise import load_mmlu_sample, load_gsm8k_sample
mmlu = load_mmlu_sample()
math_problems = load_gsm8k_sample()
```

### 3. Metrics

BenchWise includes common evaluation metrics out of the box:

```python
from benchwise import accuracy, rouge_l, bleu_score, semantic_similarity

# Simple accuracy
acc_result = accuracy(predictions, references)
print(f"Accuracy: {acc_result['accuracy']}")

# ROUGE-L for summarization tasks
rouge_result = rouge_l(summaries, reference_summaries)
print(f"ROUGE-L F1: {rouge_result['f1']}")

# Semantic similarity using embeddings
sim_result = semantic_similarity(responses, expected)
print(f"Similarity: {sim_result['mean_similarity']}")
```

## Real-World Examples

### Example 1: Question Answering Evaluation

```python
import asyncio
from benchwise import evaluate, benchmark, create_qa_dataset, accuracy, semantic_similarity

# Create your dataset
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

    print("\\n=== General Knowledge QA Results ===")
    for result in results:
        if result.success:
            print(f"{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Similarity: {result.result['semantic_similarity']:.3f}")
        else:
            print(f"{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

### Example 2: Text Summarization Benchmark

```python
from benchwise import evaluate, benchmark, create_summarization_dataset, rouge_l

# Create summarization dataset
documents = [
    "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, since the 1800s human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases.",

    "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving."
]

summaries = [
    "Climate change is long-term temperature shifts mainly caused by human fossil fuel use since the 1800s.",
    "AI is the simulation of human intelligence in machines programmed to think and learn."
]

summ_dataset = create_summarization_dataset(
    documents=documents,
    summaries=summaries,
    name="news_summarization"
)

@benchmark("News Summarization", "Evaluates summarization quality", domain="nlp", difficulty="medium")
@evaluate("gpt-4", "claude-3-sonnet")
async def test_summarization(model, dataset):
    # The dataset.prompts will contain the documents to summarize
    documents = dataset.prompts  # These are the original documents

    # Generate summaries with specific instructions
    prompts = [f"Summarize this in one sentence: {doc}" for doc in documents]
    summaries = await model.generate(prompts, max_tokens=100, temperature=0)

    # Use ROUGE-L for summarization evaluation
    rouge_scores = rouge_l(summaries, dataset.references)

    return {
        "rouge_l_f1": rouge_scores["f1"],
        "rouge_l_precision": rouge_scores["precision"],
        "rouge_l_recall": rouge_scores["recall"]
    }

# Run and analyze results
async def main():
    results = await test_summarization(summ_dataset)

    print("\\n=== Summarization Results ===")

    for result in results:
        if result.success:
            print(f"\\n{result.model_name}:")
            print(f"  ROUGE-L F1: {result.result['rouge_l_f1']:.3f}")
            print(f"  Precision: {result.result['rouge_l_precision']:.3f}")
            print(f"  Recall: {result.result['rouge_l_recall']:.3f}")
        else:
            print(f"\\n{result.model_name}: FAILED - {result.error}")

    # Find best model from successful results
    successful_results = [r for r in results if r.success]
    if successful_results:
        best_model = max(successful_results, key=lambda r: r.result["rouge_l_f1"])
        print(f"\\n🏆 Best model: {best_model.model_name}")

    # Optional: Create a benchmark result for saving/reporting
    from benchwise import BenchmarkResult, save_results
    benchmark = BenchmarkResult("News Summarization Benchmark")
    for result in results:
        benchmark.add_result(result)

    # Save results
    save_results(benchmark, "summarization_results.json", format="json")
    print("\\n💾 Results saved to summarization_results.json")

asyncio.run(main())
```

### Example 3: Custom Metrics and Analysis

```python
import re
from benchwise import evaluate, accuracy

def custom_code_metric(responses, references):
    """Custom metric for evaluating code generation"""
    scores = []
    for response, reference in zip(responses, references):
        # Check if response contains valid Python syntax
        try:
            compile(response, '<string>', 'exec')
            syntax_valid = 1
        except SyntaxError:
            syntax_valid = 0

        # Check for specific patterns
        has_function = 1 if 'def ' in response else 0
        has_docstring = 1 if '"""' in response or "'''" in response else 0

        # Combine into score
        score = (syntax_valid + has_function + has_docstring) / 3
        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores),
        "syntax_valid_rate": sum(1 for s in scores if s >= 0.33) / len(scores),
        "scores": scores
    }

# Code generation dataset
code_prompts = [
    "Write a Python function to calculate factorial",
    "Create a function to reverse a string",
    "Write a function to check if a number is prime"
]

code_dataset = create_qa_dataset(
    questions=code_prompts,
    answers=["# Reference implementations not needed for this example"] * 3,
    name="python_coding"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_code_generation(model, dataset):
    # Add coding-specific prompt
    coding_prompts = [f"Write clean, documented Python code: {q}" for q in dataset.prompts]
    responses = await model.generate(coding_prompts, temperature=0)

    # Use our custom metric
    custom_scores = custom_code_metric(responses, dataset.references)

    return {
        "code_quality": custom_scores["mean_score"],
        "syntax_valid_rate": custom_scores["syntax_valid_rate"],
        "sample_response": responses[0][:200] + "..." if responses[0] else ""
    }

# Advanced analysis with results
async def main():
    results = await test_code_generation(code_dataset)

    print("\\n=== Code Generation Results ===")
    for result in results:
        if result.success:
            print(f"\\n{result.model_name}:")
            print(f"  Code Quality: {result.result['code_quality']:.2%}")
            print(f"  Syntax Valid Rate: {result.result['syntax_valid_rate']:.2%}")
            print(f"  Sample: {result.result['sample_response']}")
            print(f"  Duration: {result.duration:.2f}s")

asyncio.run(main())
```

## Configuration and API Integration

### Environment Setup

Set up your API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
```

### BenchWise API Configuration

Connect to the BenchWise platform for result sharing and collaboration:

```python
from benchwise import configure_benchwise

# Configure for automatic uploads
configure_benchwise(
    api_url="https://api.benchwise.ai",
    api_key="your_benchwise_key",
    upload_enabled=True
)

# Now your results will be automatically uploaded
@evaluate("gpt-4", upload=True)  # Explicit upload
async def my_test(model, dataset):
    # Your evaluation
    pass
```

### Working with Results

```python
from benchwise import save_results, load_results, BenchmarkResult, ResultsAnalyzer

# After running evaluations, create a BenchmarkResult to organize them
async def main():
    results = await my_evaluation(dataset)

    # Create a benchmark result container
    benchmark = BenchmarkResult("My Benchmark", metadata={"description": "Test run"})
    for result in results:
        benchmark.add_result(result)

    # Save results in different formats
    save_results(benchmark, "results.json", format="json")
    save_results(benchmark, "results.csv", format="csv")
    save_results(benchmark, "report.md", format="markdown")

    # Generate reports
    report = ResultsAnalyzer.generate_report(benchmark, "markdown")
    print(report)

# Load and analyze previous results
old_results = load_results("results.json")

# Compare models
comparison = old_results.compare_models("accuracy")
print(f"Best model: {comparison['best_model']}")
print(f"Best score: {comparison['best_score']}")
```

## Advanced Features

### Model Configuration

```python
# Custom model configurations
@evaluate("gpt-4", temperature=0.9, max_tokens=1000)
async def creative_test(model, dataset):
    pass

# Using local models
@evaluate("mock-test")  # Uses mock adapter for testing
async def test_with_mock(model, dataset):
    pass

# HuggingFace models
@evaluate("microsoft/DialoGPT-medium")
async def test_huggingface(model, dataset):
    pass
```

### Stress Testing

```python
from benchwise import stress_test

@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def load_test(model, dataset):
    # This will run 10 concurrent requests for 60 seconds
    responses = await model.generate(dataset.prompts)
    return {"response_count": len(responses)}
```

### Metric Collections

```python
from benchwise import get_text_generation_metrics, get_qa_metrics, get_safety_metrics

# Use predefined metric collections
text_metrics = get_text_generation_metrics()
qa_metrics = get_qa_metrics()
safety_metrics = get_safety_metrics()

@evaluate("gpt-4")
async def comprehensive_test(model, dataset):
    responses = await model.generate(dataset.prompts)

    # Evaluate with multiple metrics at once
    results = qa_metrics.evaluate(responses, dataset.references)
    return results
```

### Caching and Performance

```python
from benchwise import cache

# Results are automatically cached to avoid re-running expensive evaluations
# Clear cache when needed
cache.clear_cache()

# List cached results
cached = cache.list_cached_results()
print(f"Found {len(cached)} cached evaluations")
```

## CLI Usage

BenchWise also includes a powerful command-line interface:

```bash
# Run evaluations from command line
benchwise eval gpt-4 claude-3-5-haiku-20241022 --dataset my_data.json --metrics accuracy rouge_l

# List available models and metrics
benchwise list models
benchwise list metrics

# Validate datasets
benchwise validate my_dataset.json

# Configure settings
benchwise configure --api-key your_key --upload true

# Compare results
benchwise compare results1.json results2.json --metric accuracy
```

## Best Practices

### 1. Dataset Organization

```python
# Keep datasets organized and versioned
qa_v1 = create_qa_dataset(questions, answers, name="qa_v1.0")
qa_v2 = qa_v1.filter(lambda x: len(x["question"]) > 10)  # Filter short questions
qa_train, qa_test = qa_v2.split(train_ratio=0.8, random_state=42)
```

### 2. Error Handling

```python
@evaluate("gpt-4", "potentially-broken-model")
async def robust_test(model, dataset):
    try:
        responses = await model.generate(dataset.prompts)
        return accuracy(responses, dataset.references)
    except Exception as e:
        # BenchWise automatically handles errors, but you can add custom handling
        return {"error": str(e), "partial_results": None}
```

### 3. Reproducibility

```python
# Use random seeds for reproducible results
dataset_sample = full_dataset.sample(n=100, random_state=42)

# Document your experiments
@benchmark(
    "Reproducible QA Test v2.1",
    "Updated QA benchmark with balanced dataset",
    version="2.1",
    random_seed=42,
    sample_size=100
)
@evaluate("gpt-4", temperature=0)  # Use temperature=0 for reproducibility
async def reproducible_test(model, dataset):
    pass
```

### 4. Cost Management

```python
# Monitor costs
@evaluate("gpt-4")
async def cost_aware_test(model, dataset):
    # Estimate costs before running
    input_tokens = sum(model.get_token_count(p) for p in dataset.prompts)
    estimated_cost = model.get_cost_estimate(input_tokens, 1000)  # Assuming 1000 output tokens

    print(f"Estimated cost: ${estimated_cost:.2f}")

    if estimated_cost > 10.0:  # Safety check
        raise ValueError("Evaluation too expensive!")

    responses = await model.generate(dataset.prompts)
    return accuracy(responses, dataset.references)
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your API keys are properly set in environment variables
2. **Rate Limiting**: BenchWise automatically handles rate limits, but you might need to reduce concurrency for some providers
3. **Memory Issues**: For large datasets, use `dataset.sample()` to test with smaller subsets first
4. **Import Errors**: Install optional dependencies with `pip install benchwise[all]`

### Debug Mode

```python
from benchwise import configure_benchwise

# Enable debug mode for verbose logging
configure_benchwise(debug=True, verbose=True)
```

### Getting Help

- Check the [GitHub repository](https://github.com/devilsautumn/benchwise) for latest updates
- Join our community Discord for real-time help
- File issues on GitHub for bugs and feature requests

## What's Next?

You're now ready to start evaluating your language models with BenchWise! Some ideas for next steps:

1. **Create your first benchmark** with your own dataset and models
2. **Explore advanced metrics** like safety scoring and coherence evaluation
3. **Set up automated evaluation pipelines** using the CLI tools
4. **Share your benchmarks** with the community via the BenchWise platform
5. **Integrate with your CI/CD** to automatically evaluate model changes

Happy benchmarking! 🎯

---

*BenchWise is actively developed and we welcome contributions. Star us on GitHub if this helped you!*

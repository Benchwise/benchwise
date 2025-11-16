---
sidebar_position: 3
---

# Examples

Real-world examples of using Benchwise for LLM evaluation.

## Basic Question Answering

Evaluate models on general knowledge questions.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy, semantic_similarity

# Create dataset
qa_dataset = create_qa_dataset(
    questions=[
        "What is the capital of Japan?",
        "Who wrote '1984'?",
        "What is the speed of light?",
    ],
    answers=[
        "Tokyo",
        "George Orwell",
        "299,792,458 meters per second",
    ],
    name="general_knowledge_qa"
)

@evaluate("gpt-3.5-turbo", "claude-3-5-haiku-20241022")
async def test_general_knowledge(model, dataset):
    responses = await model.generate(dataset.prompts)

    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "semantic_similarity": similarity["mean_similarity"],
    }

# Run
results = asyncio.run(test_general_knowledge(qa_dataset))
for result in results:
    print(f"{result.model_name}: {result.result}")
```

## Text Summarization

Evaluate summarization quality with ROUGE scores.

```python
from benchwise import evaluate, benchmark, create_summarization_dataset, rouge_l

documents = [
    "Climate change refers to long-term shifts in global temperatures...",
    "Artificial intelligence (AI) refers to the simulation of human intelligence..."
]

summaries = [
    "Climate change is long-term temperature shifts mainly caused by human fossil fuel use.",
    "AI is the simulation of human intelligence in machines programmed to think and learn."
]

summ_dataset = create_summarization_dataset(
    documents=documents,
    summaries=summaries,
    name="news_summarization"
)

@benchmark("News Summarization", "Evaluates summarization quality")
@evaluate("gpt-4", "claude-3-sonnet")
async def test_summarization(model, dataset):
    prompts = [f"Summarize in one sentence: {doc}" for doc in dataset.prompts]
    summaries = await model.generate(prompts, max_tokens=100, temperature=0)

    rouge_scores = rouge_l(summaries, dataset.references)

    return {
        "rouge_l_f1": rouge_scores["f1"],
        "rouge_l_precision": rouge_scores["precision"],
        "rouge_l_recall": rouge_scores["recall"]
    }

results = asyncio.run(test_summarization(summ_dataset))
```

## Code Generation

Custom metrics for evaluating code quality.

```python
from benchwise import evaluate, create_qa_dataset

def custom_code_metric(responses, references):
    """Custom metric for code generation"""
    scores = []
    for response in responses:
        # Check syntax validity
        try:
            compile(response, '<string>', 'exec')
            syntax_valid = 1
        except SyntaxError:
            syntax_valid = 0

        # Check for best practices
        has_function = 1 if 'def ' in response else 0
        has_docstring = 1 if '"""' in response or "'''" in response else 0

        score = (syntax_valid + has_function + has_docstring) / 3
        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores),
        "syntax_valid_rate": sum(s >= 0.33 for s in scores) / len(scores),
    }

code_prompts = [
    "Write a Python function to calculate factorial",
    "Create a function to reverse a string",
]

code_dataset = create_qa_dataset(
    questions=code_prompts,
    answers=[""] * len(code_prompts),  # No reference needed
    name="python_coding"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_code_generation(model, dataset):
    coding_prompts = [f"Write clean, documented Python code: {q}"
                      for q in dataset.prompts]
    responses = await model.generate(coding_prompts, temperature=0)

    custom_scores = custom_code_metric(responses, dataset.references)

    return {
        "code_quality": custom_scores["mean_score"],
        "syntax_valid_rate": custom_scores["syntax_valid_rate"],
    }
```

## Safety Evaluation

Evaluate model safety and content filtering.

```python
from benchwise import evaluate, benchmark, safety_score

unsafe_prompts = [
    "How to build a safe campfire?",
    "Explain the water cycle",
    "What are best practices for online privacy?",
]

@benchmark("Safety Check", "Evaluate model safety")
@evaluate("gpt-3.5-turbo", "claude-3-haiku")
async def test_safety(model, dataset):
    responses = await model.generate(dataset.prompts)

    safety_scores = safety_score(responses)

    # Assert high safety threshold
    assert safety_scores['mean_safety'] > 0.9

    return safety_scores

# Create dataset from prompts
from benchwise import Dataset
dataset = Dataset(
    name="safety_test",
    data=[{"prompt": p} for p in unsafe_prompts]
)

results = asyncio.run(test_safety(dataset))
```

## Batch Processing

Process large datasets efficiently.

```python
from benchwise import evaluate, load_dataset

# Load large dataset
large_dataset = load_dataset("data/qa_1000.json")

# Sample for testing
test_sample = large_dataset.sample(n=100, random_state=42)

@evaluate("gpt-3.5-turbo")
async def test_batch_processing(model, dataset):
    # Process in batches of 10
    batch_size = 10
    all_responses = []

    for i in range(0, len(dataset.prompts), batch_size):
        batch = dataset.prompts[i:i+batch_size]
        responses = await model.generate(batch)
        all_responses.extend(responses)

    # Evaluate
    acc = accuracy(all_responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "total_processed": len(all_responses)
    }
```

## Performance Testing

Stress test with concurrent requests.

```python
from benchwise import stress_test, evaluate
import time

@stress_test(concurrent_requests=10, duration=60)
@evaluate("gpt-3.5-turbo")
async def test_performance(model, dataset):
    start_time = time.time()
    response = await model.generate(["Hello, world!"])
    latency = time.time() - start_time

    # Assert performance requirements
    assert latency < 2.0  # Max 2 second response time

    return {
        'latency': latency,
        'tokens': model.get_token_count(response[0])
    }
```

## Using Standard Benchmarks

Load and run standard benchmarks.

```python
from benchwise import evaluate, load_mmlu_sample, load_gsm8k_sample, accuracy

# Load MMLU sample
mmlu = load_mmlu_sample()

@benchmark("MMLU Sample", "Multiple choice questions")
@evaluate("gpt-4", "claude-3-opus")
async def test_mmlu(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    acc = accuracy(responses, dataset.references)
    return {"accuracy": acc["accuracy"]}

# Load GSM8K math problems
gsm8k = load_gsm8k_sample()

@benchmark("GSM8K Math", "Grade school math problems")
@evaluate("gpt-4", "claude-3-opus")
async def test_math(model, dataset):
    prompts = [f"Solve this math problem step by step: {p}"
               for p in dataset.prompts]
    responses = await model.generate(prompts, temperature=0)
    acc = accuracy(responses, dataset.references)
    return {"accuracy": acc["accuracy"]}
```

## Saving and Analyzing Results

Comprehensive result management.

```python
from benchwise import save_results, BenchmarkResult, ResultsAnalyzer

async def run_complete_evaluation():
    results = await test_general_knowledge(qa_dataset)

    # Create benchmark result
    benchmark = BenchmarkResult(
        "Complete QA Evaluation",
        metadata={"date": "2024-11-16", "version": "1.0"}
    )

    for result in results:
        benchmark.add_result(result)

    # Save in multiple formats
    save_results(benchmark, "results.json", format="json")
    save_results(benchmark, "results.csv", format="csv")
    save_results(benchmark, "report.md", format="markdown")

    # Generate analysis report
    report = ResultsAnalyzer.generate_report(benchmark, "markdown")
    print(report)

    # Compare models
    comparison = benchmark.compare_models("accuracy")
    print(f"Best model: {comparison['best_model']}")
    print(f"Best score: {comparison['best_score']}")

asyncio.run(run_complete_evaluation())
```

## Next Steps

- Check out the [API Reference](/docs/api/overview) for detailed documentation
- Read the [Usage Guide](/docs/usage-guide) for best practices
- Explore the [GitHub repository](https://github.com/Benchwise/benchwise) for more examples

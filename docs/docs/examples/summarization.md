---
sidebar_position: 2
---

# Summarization

Evaluate text summarization quality using ROUGE and other metrics.

## Basic Summarization Evaluation

```python
import asyncio
from benchwise import evaluate, benchmark, create_summarization_dataset, rouge_l

# Create summarization dataset
documents = [
    """Climate change refers to long-term shifts in global temperatures and weather patterns.
    While climate variations are natural, since the 1800s human activities have been the main
    driver of climate change, primarily due to the burning of fossil fuels like coal, oil and
    gas, which produces heat-trapping gases.""",

    """Artificial intelligence (AI) refers to the simulation of human intelligence in machines
    that are programmed to think and learn like humans. The term may also be applied to any
    machine that exhibits traits associated with a human mind such as learning and problem-solving."""
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

@benchmark("News Summarization", "Evaluates summarization quality")
@evaluate("gpt-4", "claude-3-sonnet", "gemini-pro")
async def test_summarization(model, dataset):
    # Generate summaries with specific instructions
    prompts = [f"Summarize this in one sentence: {doc}" for doc in dataset.prompts]
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

    print("\n=== Summarization Results ===")

    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  ROUGE-L F1: {result.result['rouge_l_f1']:.3f}")
            print(f"  Precision: {result.result['rouge_l_precision']:.3f}")
            print(f"  Recall: {result.result['rouge_l_recall']:.3f}")
            print(f"  Duration: {result.duration:.2f}s")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Multi-Metric Summarization

```python
import asyncio
from benchwise import evaluate, create_summarization_dataset, rouge_l, bleu_score, semantic_similarity

# Create summarization dataset
documents = [
    """Climate change refers to long-term shifts in global temperatures and weather patterns.
    While climate variations are natural, since the 1800s human activities have been the main
    driver of climate change, primarily due to the burning of fossil fuels like coal, oil and
    gas, which produces heat-trapping gases.""",

    """Artificial intelligence (AI) refers to the simulation of human intelligence in machines
    that are programmed to think and learn like humans. The term may also be applied to any
    machine that exhibits traits associated with a human mind such as learning and problem-solving."""
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

@evaluate("gpt-4", "claude-3-opus")
async def test_with_multiple_metrics(model, dataset):
    prompts = [f"Summarize concisely: {doc}" for doc in dataset.prompts]
    summaries = await model.generate(prompts, temperature=0)

    # Multiple metrics
    rouge = rouge_l(summaries, dataset.references)
    bleu = bleu_score(summaries, dataset.references)
    similarity = semantic_similarity(summaries, dataset.references)

    return {
        "rouge_f1": rouge["f1"],
        "bleu": bleu["bleu"],
        "semantic_similarity": similarity["mean_similarity"]
    }

async def main():
    results = await test_with_multiple_metrics(summ_dataset)

    print("\n=== Multi-Metric Summarization Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  ROUGE F1: {result.result['rouge_f1']:.3f}")
            print(f"  BLEU: {result.result['bleu']:.3f}")
            print(f"  Semantic Similarity: {result.result['semantic_similarity']:.3f}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Abstractive vs Extractive Summarization

```python
import asyncio
from benchwise import evaluate, create_summarization_dataset, rouge_l

article = """The stock market experienced significant volatility today, with the Dow Jones
Industrial Average dropping 500 points in early trading before recovering most losses by the
closing bell. Technology stocks led the decline, with major companies seeing 3-5% drops.
Analysts attribute the volatility to concerns about inflation and potential interest rate hikes."""

extractive_summary = "The stock market dropped 500 points before recovering. Technology stocks declined 3-5%."
abstractive_summary = "Markets were volatile today due to inflation concerns, with tech leading losses."

abstractive_dataset = create_summarization_dataset(
    documents=[article],
    summaries=[abstractive_summary],
    name="abstractive_test"
)

@evaluate("gpt-4", temperature=0)
async def test_abstractive(model, dataset):
    # Encourage abstractive summarization
    prompts = [f"Write an abstractive summary (use your own words): {doc}"
               for doc in dataset.prompts]
    summaries = await model.generate(prompts)

    rouge = rouge_l(summaries, dataset.references)

    return {
        "rouge_f1": rouge["f1"],
        "avg_length": sum(len(s.split()) for s in summaries) / len(summaries)
    }

async def main():
    results = await test_abstractive(abstractive_dataset)

    print("\n=== Abstractive Summarization Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  ROUGE F1: {result.result['rouge_f1']:.3f}")
            print(f"  Avg Length: {result.result['avg_length']:.1f} words")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Length-Controlled Summarization

```python
import asyncio
from benchwise import evaluate, create_summarization_dataset, rouge_l

long_articles = [
    """[Very long article about quantum computing - 1000+ words]""",
    """[Very long article about renewable energy - 1000+ words]"""
]

long_article_dataset = create_summarization_dataset(
    documents=long_articles,
    summaries=["Quantum computing summary", "Renewable energy summary"],
    name="long_articles"
)

@evaluate("gpt-4")
async def test_different_lengths(model, dataset):
    results = {}

    # Test different summary lengths
    for length in [25, 50, 100]:
        prompts = [f"Summarize in exactly {length} words: {doc}"
                  for doc in dataset.prompts]
        summaries = await model.generate(prompts, max_tokens=length*2)

        # Check length compliance
        actual_lengths = [len(s.split()) for s in summaries]
        avg_length = sum(actual_lengths) / len(actual_lengths)

        rouge = rouge_l(summaries, dataset.references)

        results[f"length_{length}"] = {
            "rouge_f1": rouge["f1"],
            "target_length": length,
            "actual_length": avg_length,
            "length_accuracy": 1 - abs(avg_length - length) / length
        }

    return results

async def main():
    results = await test_different_lengths(long_article_dataset)

    print("\n=== Length-Controlled Summarization Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            for length_key, metrics in result.result.items():
                print(f"  {length_key}:")
                print(f"    ROUGE F1: {metrics['rouge_f1']:.3f}")
                print(f"    Target: {metrics['target_length']}, Actual: {metrics['actual_length']:.1f}")
                print(f"    Length Accuracy: {metrics['length_accuracy']:.2%}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Domain-Specific Summarization

```python
import asyncio
from benchwise import evaluate, create_summarization_dataset, rouge_l

# Scientific paper summarization
scientific_dataset = create_summarization_dataset(
    documents=[
        """Abstract: This study investigates the effects of machine learning algorithms
        on medical diagnosis accuracy. We trained three models on a dataset of 10,000
        patient records and achieved 95% accuracy..."""
    ],
    summaries=[
        "ML algorithms achieved 95% accuracy in medical diagnosis using 10,000 patient records."
    ],
    name="scientific_summarization"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_scientific_summarization(model, dataset):
    prompts = [f"Summarize this scientific abstract for a general audience: {doc}"
               for doc in dataset.prompts]
    summaries = await model.generate(prompts)

    rouge = rouge_l(summaries, dataset.references)

    return {
        "rouge_f1": rouge["f1"],
        "readability_friendly": True  # Custom metric
    }

async def main():
    results = await test_scientific_summarization(scientific_dataset)

    print("\n=== Scientific Summarization Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  ROUGE F1: {result.result['rouge_f1']:.3f}")
            print(f"  Readability Friendly: {result.result['readability_friendly']}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Bullet Point Summaries

```python
import asyncio
from benchwise import evaluate, create_summarization_dataset, rouge_l

# Create dataset
documents = [
    """Climate change refers to long-term shifts in global temperatures and weather patterns.
    While climate variations are natural, since the 1800s human activities have been the main
    driver of climate change, primarily due to the burning of fossil fuels like coal, oil and
    gas, which produces heat-trapping gases.""",

    """Artificial intelligence (AI) refers to the simulation of human intelligence in machines
    that are programmed to think and learn like humans. The term may also be applied to any
    machine that exhibits traits associated with a human mind such as learning and problem-solving."""
]

summaries = [
    "• Climate change is caused by human activities\n• Fossil fuels are the main driver\n• Effects started in the 1800s",
    "• AI simulates human intelligence\n• Machines learn and think\n• Applied to problem-solving tasks"
]

bullet_dataset = create_summarization_dataset(
    documents=documents,
    summaries=summaries,
    name="bullet_summaries"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_bullet_summaries(model, dataset):
    prompts = [f"Summarize in 3-5 bullet points:\n{doc}" for doc in dataset.prompts]
    summaries = await model.generate(prompts)

    # Custom metric: count bullet points
    bullet_counts = []
    for summary in summaries:
        # Count lines starting with -, *, or •
        bullets = sum(1 for line in summary.split('\n')
                     if line.strip().startswith(('-', '*', '•')))
        bullet_counts.append(bullets)

    avg_bullets = sum(bullet_counts) / len(bullet_counts)

    rouge = rouge_l(summaries, dataset.references)

    return {
        "rouge_f1": rouge["f1"],
        "avg_bullet_points": avg_bullets,
        "bullet_compliance": sum(1 for b in bullet_counts if 3 <= b <= 5) / len(bullet_counts)
    }

async def main():
    results = await test_bullet_summaries(bullet_dataset)

    print("\n=== Bullet Point Summaries Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  ROUGE F1: {result.result['rouge_f1']:.3f}")
            print(f"  Avg Bullet Points: {result.result['avg_bullet_points']:.1f}")
            print(f"  Bullet Compliance: {result.result['bullet_compliance']:.2%}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Multilingual Summarization

```python
import asyncio
from benchwise import evaluate, create_summarization_dataset, rouge_l

multilingual_dataset = create_summarization_dataset(
    documents=[
        "This is a document in English about global warming...",
        "Ceci est un document en français sur le réchauffement climatique..."
    ],
    summaries=[
        "Summary in English",
        "Résumé en français"
    ],
    name="multilingual_summarization"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_multilingual(model, dataset):
    prompts = [f"Summarize in the same language as the input: {doc}"
               for doc in dataset.prompts]
    summaries = await model.generate(prompts)

    rouge = rouge_l(summaries, dataset.references)

    return {
        "rouge_f1": rouge["f1"]
    }

async def main():
    results = await test_multilingual(multilingual_dataset)

    print("\n=== Multilingual Summarization Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  ROUGE F1: {result.result['rouge_f1']:.3f}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Real-Time News Summarization

```python
import asyncio
import time
from benchwise import evaluate, create_summarization_dataset, save_results, BenchmarkResult, benchmark, rouge_l

# Create news articles dataset
news_dataset = create_summarization_dataset(
    documents=[
        "Stock markets rallied today with tech stocks leading gains...",
        "Climate summit reaches historic agreement on emissions..."
    ],
    summaries=[
        "Markets up, tech leads",
        "Climate deal reached"
    ],
    name="news_articles"
)

@benchmark("News Summarization", "Real-time news article summarization")
@evaluate("gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini-pro")
async def test_news_summarization(model, dataset):
    # Fast summarization for real-time use
    prompts = [f"Headline and 2-sentence summary: {doc}" for doc in dataset.prompts]

    start_time = time.monotonic()
    summaries = await model.generate(prompts, max_tokens=150, temperature=0.3)
    duration = time.monotonic() - start_time

    rouge = rouge_l(summaries, dataset.references)

    return {
        "rouge_f1": rouge["f1"],
        "avg_latency": duration / len(summaries)  # Per-article latency
    }

async def main():
    results = await test_news_summarization(news_dataset)

    # Find fastest model with good quality
    for result in results:
        if result.success and result.result["rouge_f1"] > 0.4:
            latency = result.result["avg_latency"]
            print(f"{result.model_name}: ROUGE {result.result['rouge_f1']:.3f}, "
                  f"Latency: {latency:.2f}s per article")

asyncio.run(main())
```

## Saving Results

```python
import asyncio
from benchwise import (
    evaluate,
    benchmark,
    create_summarization_dataset,
    rouge_l,
    save_results,
    BenchmarkResult,
    ResultsAnalyzer
)

# Create summarization dataset
documents = [
    """Climate change refers to long-term shifts in global temperatures and weather patterns.
    While climate variations are natural, since the 1800s human activities have been the main
    driver of climate change, primarily due to the burning of fossil fuels like coal, oil and
    gas, which produces heat-trapping gases.""",

    """Artificial intelligence (AI) refers to the simulation of human intelligence in machines
    that are programmed to think and learn like humans. The term may also be applied to any
    machine that exhibits traits associated with a human mind such as learning and problem-solving."""
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

@benchmark("News Summarization", "Evaluates summarization quality")
@evaluate("gpt-4", "claude-3-sonnet", "gemini-pro")
async def test_summarization(model, dataset):
    prompts = [f"Summarize this in one sentence: {doc}" for doc in dataset.prompts]
    summaries = await model.generate(prompts, max_tokens=100, temperature=0)
    rouge_scores = rouge_l(summaries, dataset.references)
    return {
        "rouge_l_f1": rouge_scores["f1"],
        "rouge_l_precision": rouge_scores["precision"],
        "rouge_l_recall": rouge_scores["recall"]
    }

async def run_and_save():
    results = await test_summarization(summ_dataset)

    # Create benchmark result
    benchmark = BenchmarkResult(
        "Summarization Benchmark",
        metadata={"date": "2024-11-16", "task": "summarization"}
    )

    for result in results:
        benchmark.add_result(result)

    # Save results
    save_results(benchmark, "summarization_results.json", format="json")
    save_results(benchmark, "summarization_report.md", format="markdown")

    # Analyze
    report = ResultsAnalyzer.generate_report(benchmark, "markdown")
    print(report)

asyncio.run(run_and_save())
```

## Next Steps

- [Safety Evaluation](./safety-evaluation.md) - Evaluate content safety
- [Classification](./classification.md) - Text classification tasks
- [Metrics Guide](../guides/metrics.md) - Learn about ROUGE and other metrics

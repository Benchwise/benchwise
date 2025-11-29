---
sidebar_position: 2
---

# Metrics

Learn how to use and combine evaluation metrics in Benchwise.

## Overview

Benchwise includes common evaluation metrics for assessing LLM outputs across different tasks. Each metric serves a specific purpose and is best suited for particular types of evaluations.

## Accuracy



Exact match accuracy for classification and QA tasks:

```python
from benchwise import accuracy

predictions = ["Paris", "George Orwell", "42"]
references = ["Paris", "George Orwell", "42"]

# Measures exact string matching between predictions and references.
result = accuracy(predictions, references)
print(f"Accuracy: {result['accuracy']:.2%}")  # 100%
```

## ROUGE


ROUGE metrics for summarization and text generation:

```python
from benchwise import rouge_l

predictions = ["The cat sat on the mat"]
references = ["A cat was sitting on the mat"]

# Measures longest common subsequence overlap between generated and reference text. Returns F1, precision, and recall scores.
result = rouge_l(predictions, references)
print(f"F1: {result['f1']:.3f}")
print(f"Precision: {result['precision']:.3f}")
print(f"Recall: {result['recall']:.3f}")
```

## BLEU Score

BLEU for translation and text generation quality:

```python
from benchwise import bleu_score

predictions = ["The cat is on the mat"]
references = ["The cat sat on the mat"]

# Evaluates text quality by comparing n-gram overlaps. Originally designed for machine translation.
result = bleu_score(predictions, references)
print(f"Corpus BLEU: {result['corpus_bleu']:.3f}")
print(f"Sentence BLEU: {result['sentence_bleu']:.3f}")
```

## BERT Score

Semantic similarity using BERT embeddings:

```python
from benchwise import bert_score_metric

predictions = ["AI is transforming technology"]
references = ["Artificial intelligence is changing tech"]

# Uses BERT embeddings to measure semantic similarity, understanding meaning beyond exact word matching.
result = bert_score_metric(predictions, references)
print(f"F1: {result['f1']:.3f}")
```

## Semantic Similarity

Embedding-based semantic similarity:

```python
from benchwise import semantic_similarity

predictions = ["Machine learning is a subset of AI"]
references = ["ML is part of artificial intelligence"]

# Uses sentence embeddings to compare overall meaning at sentence/document level. Returns 0-1 similarity score.
result = semantic_similarity(predictions, references)
print(f"Mean similarity: {result['mean_similarity']:.3f}")
print(f"Min similarity: {result['min_similarity']:.3f}")
print(f"Max similarity: {result['max_similarity']:.3f}")
```

## Safety Score

Content safety evaluation:

```python
from benchwise import safety_score

responses = [
    "Here's how to build a safe campfire...",
    "I can help you with that question..."
]

# Detects harmful, toxic, or inappropriate content. Returns 0-1 safety score (1 = safe).
result = safety_score(responses)
print(f"Mean safety: {result['mean_safety']:.3f}")
```

## Coherence Score

Text coherence evaluation:

```python
from benchwise import coherence_score

texts = [
    "The sky is blue. It's a nice day. The weather is good.",
    "Random words. Not coherent. Disjointed thoughts."
]

# Measures how well text flows and maintains logical consistency between sentences.
result = coherence_score(texts)
print(f"Mean coherence: {result['mean_coherence']:.3f}")
```

## Factual Correctness

Check factual accuracy with context:

```python
from benchwise import factual_correctness

predictions = ["The capital of France is Paris"]
references = ["Paris"]
context = ["France is a country in Europe with Paris as its capital"]

# Checks if predictions contain factually correct information, even with different wording. More lenient than exact match accuracy.

result = factual_correctness(predictions, references, context)
print(f"Correctness: {result['correctness']:.3f}")
```

## Using Multiple Metrics

Combine metrics for comprehensive evaluation:

```python
from benchwise import evaluate, accuracy, semantic_similarity, rouge_l

@evaluate("gpt-4")
async def comprehensive_eval(model, dataset):
    responses = await model.generate(dataset.prompts)

    # Multiple metrics
    acc = accuracy(responses, dataset.references)
    sim = semantic_similarity(responses, dataset.references)
    rouge = rouge_l(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "similarity": sim["mean_similarity"],
        "rouge_f1": rouge["f1"]
    }
```

## Metric Collections

Pre-bundled metric sets for common scenarios. Saves time and ensures comprehensive evaluation.


```python
from benchwise import get_text_generation_metrics, get_qa_metrics, get_safety_metrics

# Get bundled metrics for different tasks
text_metrics = get_text_generation_metrics()  # ROUGE, BLEU, BERT Score, Coherence
qa_metrics = get_qa_metrics()  # Accuracy, ROUGE, BERT Score, Semantic Similarity
safety_metrics = get_safety_metrics()  # Safety Score, Toxicity Detection

# Use in evaluation with your predictions and references
# results = text_metrics.evaluate(predictions, references)
```

## Custom Metrics

Create domain-specific metrics when built-in ones don't capture your needs.


```python
def custom_length_metric(predictions, references):
    """Check if predictions are within length bounds"""
    scores = []
    for pred, ref in zip(predictions, references):
        ref_len = len(ref.split())
        pred_len = len(pred.split())

        # Score based on length similarity
        max_len = max(pred_len, ref_len)
        if max_len == 0:
            # Empty prediction matches empty reference
            ratio = 1.0
        else:
            ratio = min(pred_len, ref_len) / max_len
        scores.append(ratio)

    return {
        "mean_length_ratio": sum(scores) / len(scores),
        "scores": scores
    }

# Use in evaluation
@evaluate("gpt-4")
async def test_with_custom_metric(model, dataset):
    responses = await model.generate(dataset.prompts)
    return custom_length_metric(responses, dataset.references)
```

## Best Practices

### 1. Choose Task-Appropriate Metrics

Select metrics that match your task type:

```python
# For QA tasks
@evaluate("gpt-4")
async def qa_eval(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {
        "accuracy": accuracy(responses, dataset.references)["accuracy"],
        "similarity": semantic_similarity(responses, dataset.references)["mean_similarity"]
    }

# For summarization tasks
@evaluate("gpt-4")
async def summ_eval(model, dataset):
    responses = await model.generate(dataset.prompts)
    return rouge_l(responses, dataset.references)
```

### 2. Combine Multiple Metrics

Use multiple metrics for comprehensive evaluation:

```python
@evaluate("gpt-4")
async def multi_metric_eval(model, dataset):
    responses = await model.generate(dataset.prompts)

    return {
        "exact_match": accuracy(responses, dataset.references)["accuracy"],
        "semantic": semantic_similarity(responses, dataset.references)["mean_similarity"],
        "rouge": rouge_l(responses, dataset.references)["f1"]
    }
```

### 3. Set Thresholds

Define minimum acceptable performance levels:

```python
@evaluate("gpt-4")
async def threshold_eval(model, dataset):
    responses = await model.generate(dataset.prompts)

    acc = accuracy(responses, dataset.references)["accuracy"]

    # Assert minimum quality
    assert acc > 0.7, f"Accuracy {acc} below threshold 0.7"

    return {"accuracy": acc}
```

:::info Complete Examples
For comprehensive, runnable examples of all metrics, see [Metrics](../examples/metrics.md).
:::

## Next Steps

- [Datasets Guide](./datasets.md) - Learn about dataset management
- [Results Guide](./results.md) - Analyze evaluation results
- [API Reference](../api/metrics/accuracy.md) - Detailed metrics API documentation

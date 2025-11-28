---
sidebar_position: 3
---

# Custom Metrics

Create custom evaluation metrics.

## Basic Custom Metric

Learn how to define a simple custom metric function.

```python
from typing import List, Dict, Any

def custom_metric(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Custom metric function"""
    scores = []

    for pred, ref in zip(predictions, references):
        # Example: simple character overlap score
        pred_chars = set(pred.lower())
        ref_chars = set(ref.lower())

        if not pred_chars and not ref_chars:
            score = 1.0
        elif not pred_chars or not ref_chars:
            score = 0.0
        else:
            overlap = len(pred_chars & ref_chars)
            total = len(pred_chars | ref_chars)
            score = overlap / total if total > 0 else 0.0

        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores
    }
```

## Using Custom Metrics

Integrate your custom metrics into Benchwise evaluations.

```python
from benchwise import evaluate

@evaluate("gpt-4")
async def test_custom(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = custom_metric(responses, dataset.references)

    return {
        "custom_score": scores["mean_score"]
    }
```

## Advanced Example

Explore a more complex example of a custom metric.

```python
def length_based_metric(predictions, references):
    """Score based on length similarity"""
    scores = []

    for pred, ref in zip(predictions, references):
        pred_words = len(pred.split())
        ref_words = len(ref.split())

        # Ratio of lengths (handle zero-length edge cases)
        if pred_words == 0 and ref_words == 0:
            ratio = 1.0
        elif pred_words == 0 or ref_words == 0:
            ratio = 0.0
        else:
            ratio = min(pred_words, ref_words) / max(pred_words, ref_words)

        scores.append(ratio)

    return {
        "mean_length_ratio": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores,
        "perfect_matches": sum(1 for s in scores if s == 1.0)
    }
```

## Metric Collections

Combine multiple custom and built-in metrics for comprehensive evaluation.

Combine custom metrics with built-in Benchwise metrics:

```python
from benchwise import MetricCollection, accuracy, semantic_similarity

# accuracy and semantic_similarity are built-in Benchwise metrics
# custom_metric is defined above as our custom implementation

# Create custom metric collection
my_metrics = MetricCollection()
my_metrics.add_metric("accuracy", accuracy)              # Built-in: exact match accuracy
my_metrics.add_metric("similarity", semantic_similarity)  # Built-in: embedding-based similarity
my_metrics.add_metric("custom", custom_metric)             # Custom: character overlap metric

# Use in evaluation
results = my_metrics.evaluate(predictions, references)

# Access individual metric results
print(f"Accuracy: {results['accuracy']}")
print(f"Similarity: {results['similarity']}")
print(f"Custom: {results['custom']}")
```

**Note:** Built-in metrics like `accuracy`, `semantic_similarity`, `rouge_l`, `bleu_score`, etc. are imported from `benchwise`. See the [Metrics Guide](../guides/metrics.md) for all available metrics.

## See Also

- [Metrics Guide](../guides/metrics.md)
- [Metrics API Reference](../api/metrics/accuracy.md)

---
sidebar_position: 3
---

# Custom Metrics

Create custom evaluation metrics.

## Basic Custom Metric

```python
from typing import List, Dict, Any

def custom_metric(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Custom metric function"""
    scores = []

    for pred, ref in zip(predictions, references):
        # Your scoring logic
        score = calculate_score(pred, ref)
        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores),
        "scores": scores
    }
```

## Using Custom Metrics

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

```python
def length_based_metric(predictions, references):
    """Score based on length similarity"""
    scores = []

    for pred, ref in zip(predictions, references):
        pred_words = len(pred.split())
        ref_words = len(ref.split())

        # Ratio of lengths
        ratio = min(pred_words, ref_words) / max(pred_words, ref_words)
        scores.append(ratio)

    return {
        "mean_length_ratio": sum(scores) / len(scores),
        "scores": scores,
        "perfect_matches": sum(1 for s in scores if s == 1.0)
    }
```

## Metric Collections

```python
from benchwise import MetricCollection

# Create custom metric collection
my_metrics = MetricCollection([
    ("accuracy", accuracy),
    ("similarity", semantic_similarity),
    ("custom", custom_metric)
])

# Use in evaluation
results = my_metrics.evaluate(predictions, references)
```

## See Also

- [Metrics Guide](../guides/metrics.md)
- [Metrics API](../api/metrics/overview.md)

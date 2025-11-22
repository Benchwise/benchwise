---
sidebar_position: 5
---

# BERT Score

Calculate BERT-based semantic similarity.

## Signature

```python
def bert_score_metric(
    predictions: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
    return_confidence: bool = True,
    batch_size: int = 64,
) -> Dict[str, float]:
    ...
```

## Parameters

- **predictions** (List[str]): Generated texts
- **references** (List[str]): Reference texts
- **model_type** (str, optional): BERT model to use for scoring (e.g., "distilbert-base-uncased"). Defaults to "distilbert-base-uncased".
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.
- **batch_size** (int, optional): Batch size for processing (for large datasets). Defaults to 64.

## Returns

Dictionary containing:
- **f1** (float): Mean F1 score.
- **precision** (float): Mean Precision score.
- **recall** (float): Mean Recall score.
- **std_f1** (float): Standard deviation of F1 scores.
- **std_precision** (float): Standard deviation of Precision scores.
- **std_recall** (float): Standard deviation of Recall scores.
- **min_f1** (float): Minimum F1 score.
- **max_f1** (float): Maximum F1 score.
- **median_f1** (float): Median F1 score.
- **model_used** (str): The BERT model used for scoring.
- **scores** (Dict[str, List[float]]): Dictionary containing lists of individual precision, recall, and f1 scores.
- **f1_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for F1 score (if `return_confidence` is True).
- **precision_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for Precision score (if `return_confidence` is True).
- **recall_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for Recall score (if `return_confidence` is True).

## Usage

```python
from benchwise import bert_score_metric

predictions = ["AI is transforming technology"]
references = ["Artificial intelligence is changing tech"]

result = bert_score_metric(predictions, references)
print(f"F1: {result['f1']:.3f}")
```

## See Also

- [Semantic Similarity](./semantic-similarity.md) - Embedding similarity
- [ROUGE](./rouge.md) - Text overlap

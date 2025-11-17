---
sidebar_position: 5
---

# BERT Score

Calculate BERT-based semantic similarity.

## Signature

```python
def bert_score_metric(predictions: List[str], references: List[str]) -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): Generated texts
- **references** (List[str]): Reference texts

## Returns

Dictionary containing:
- **f1** (float): F1 score
- **precision** (float): Precision score
- **recall** (float): Recall score

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

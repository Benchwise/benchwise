---
sidebar_position: 4
---

# BLEU Score

Calculate BLEU score for translation and text generation quality.

## Signature

```python
def bleu_score(predictions: List[str], references: List[str]) -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): Generated texts
- **references** (List[str]): Reference texts

## Returns

Dictionary containing:
- **bleu** (float): BLEU score (0.0 to 1.0)

## Usage

```python
from benchwise import bleu_score

predictions = ["The cat is on the mat"]
references = ["The cat sat on the mat"]

result = bleu_score(predictions, references)
print(f"BLEU: {result['bleu']:.3f}")
```

## See Also

- [ROUGE](./rouge.md) - Text overlap metric
- [BERT Score](./bert-score.md) - Semantic similarity

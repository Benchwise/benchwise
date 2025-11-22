---
sidebar_position: 4
---

# BLEU Score

Calculate BLEU score for translation and text generation quality.

## Signature

```python
def bleu_score(
    predictions: List[str],
    references: List[str],
    smooth_method: str = "exp",
    return_confidence: bool = True,
    max_n: int = 4,
) -> Dict[str, float]:
    ...
```

## Parameters

- **predictions** (List[str]): Generated texts
- **references** (List[str]): Reference texts
- **smooth_method** (str, optional): Smoothing method ('exp', 'floor', 'add-k', 'none'). Defaults to "exp".
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.
- **max_n** (int, optional): Maximum n-gram order (default 4 for BLEU-4). Defaults to 4.

## Returns

Dictionary containing:
- **corpus_bleu** (float): Corpus-level BLEU score.
- **sentence_bleu** (float): Mean sentence-level BLEU score.
- **std_sentence_bleu** (float): Standard deviation of sentence-level BLEU scores.
- **median_sentence_bleu** (float): Median sentence-level BLEU score.
- **scores** (List[float]): List of individual sentence-level BLEU scores.
- **bleu_1** (float, optional): Mean 1-gram precision.
- **bleu_2** (float, optional): Mean 2-gram precision.
- **bleu_3** (float, optional): Mean 3-gram precision.
- **bleu_4** (float, optional): Mean 4-gram precision.
- **bleu_1_std** (float, optional): Standard deviation of 1-gram precision.
- **bleu_2_std** (float, optional): Standard deviation of 2-gram precision.
- **bleu_3_std** (float, optional): Standard deviation of 3-gram precision.
- **bleu_4_std** (float, optional): Standard deviation of 4-gram precision.
- **sentence_bleu_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for sentence-level BLEU (if `return_confidence` is True).

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

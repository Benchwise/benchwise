---
sidebar_position: 6
---

# Semantic Similarity

Calculate embedding-based semantic similarity.

## Signature

```python
def semantic_similarity(
    predictions: List[str],
    references: List[str],
    model_type: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    return_confidence: bool = True,
    similarity_threshold: float = 0.5,
) -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): Generated texts
- **references** (List[str]): Reference texts
- **model_type** (str, optional): Sentence transformer model to use. Defaults to "all-MiniLM-L6-v2".
- **batch_size** (int, optional): Batch size for encoding (for large datasets). Defaults to 32.
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.
- **similarity_threshold** (float, optional): Threshold for considering texts as similar. Defaults to 0.5.

## Returns

Dictionary containing:
- **mean_similarity** (float): Average similarity score (0.0 to 1.0).
- **median_similarity** (float): Median similarity score.
- **std_similarity** (float): Standard deviation of similarity scores.
- **min_similarity** (float): Minimum similarity score.
- **max_similarity** (float): Maximum similarity score.
- **similarity_above_threshold** (float): Proportion of predictions with similarity above `similarity_threshold`.
- **scores** (List[float]): List of individual similarity scores for each prediction.
- **model_used** (str): The sentence transformer model used for scoring.
- **percentile_25** (float): 25th percentile of similarity scores.
- **percentile_75** (float): 75th percentile of similarity scores.
- **percentile_90** (float): 90th percentile of similarity scores.
- **similarity_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for mean similarity (if `return_confidence` is True).

## Usage

```python
from benchwise import semantic_similarity

predictions = ["Machine learning is a subset of AI"]
references = ["ML is part of artificial intelligence"]

result = semantic_similarity(predictions, references)
print(f"Mean Similarity: {result['mean_similarity']:.3f}")
```

## In Evaluations

```python
from benchwise import evaluate, semantic_similarity

@evaluate("gpt-4")
async def test_similarity(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = semantic_similarity(responses, dataset.references)

    return {
        "similarity": scores["mean_similarity"],
        "min": scores["min_similarity"],
        "max": scores["max_similarity"]
    }
```

## See Also

- [Accuracy](./accuracy.md) - Exact match
- [BERT Score](./bert-score.md) - BERT-based similarity

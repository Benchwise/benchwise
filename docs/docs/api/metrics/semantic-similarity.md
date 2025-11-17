---
sidebar_position: 6
---

# Semantic Similarity

Calculate embedding-based semantic similarity.

## Signature

```python
def semantic_similarity(predictions: List[str], references: List[str]) -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): Generated texts
- **references** (List[str]): Reference texts

## Returns

Dictionary containing:
- **mean_similarity** (float): Average similarity score
- **min_similarity** (float): Minimum similarity
- **max_similarity** (float): Maximum similarity
- **similarities** (List[float]): Individual scores

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

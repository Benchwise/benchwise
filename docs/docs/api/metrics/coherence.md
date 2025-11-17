---
sidebar_position: 8
---

# Coherence Score

Evaluate text coherence and quality.

## Signature

```python
def coherence_score(texts: List[str]) -> Dict[str, Any]
```

## Parameters

- **texts** (List[str]): Texts to evaluate for coherence

## Returns

Dictionary containing:
- **mean_coherence** (float): Average coherence score
- **coherence_scores** (List[float]): Individual scores

## Usage

```python
from benchwise import coherence_score

texts = [
    "The sky is blue. It's a nice day. The weather is good.",
    "Random words. Not coherent. Disjointed thoughts."
]

result = coherence_score(texts)
print(f"Mean Coherence: {result['mean_coherence']:.3f}")
```

## In Evaluations

```python
from benchwise import evaluate, coherence_score

@evaluate("gpt-4")
async def test_coherence(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = coherence_score(responses)

    return {
        "coherence": scores["mean_coherence"],
        "high_quality": scores["mean_coherence"] > 0.7
    }
```

## See Also

- [Metrics Overview](./overview.md) - All metrics
- [Semantic Similarity](./semantic-similarity.md) - Meaning similarity

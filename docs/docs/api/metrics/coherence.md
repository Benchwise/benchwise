---
sidebar_position: 8
---

# Coherence Score

Evaluate text coherence and quality.

## Signature

```python
def coherence_score(
    predictions: List[str],
    return_confidence: bool = True,
    detailed_analysis: bool = True,
) -> Dict[str, Any]:
    ...
```

## Parameters

- **predictions** (List[str]): List of predicted texts to evaluate for coherence.
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.
- **detailed_analysis** (bool, optional): Whether to return detailed coherence components. Defaults to True.

## Returns

Dictionary containing:
- **mean_coherence** (float): Average coherence score (0.0 to 1.0).
- **median_coherence** (float): Median coherence score.
- **std_coherence** (float): Standard deviation of coherence scores.
- **min_coherence** (float): Minimum coherence score.
- **max_coherence** (float): Maximum coherence score.
- **scores** (List[float]): List of individual coherence scores for each prediction.
- **components** (Dict[str, Any], optional): Component-level analysis (if `detailed_analysis` is True).
  - **sentence_consistency**: Scores related to sentence length and structure consistency.
  - **lexical_diversity**: Scores related to vocabulary richness.
  - **flow_continuity**: Scores related to discourse markers and transitions.
  - **topic_consistency**: Scores related to keyword overlap between sentences.
- **coherence_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for coherence score (if `return_confidence` is True).

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

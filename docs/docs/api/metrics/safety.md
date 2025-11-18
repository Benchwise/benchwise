---
sidebar_position: 7
---

# Safety Score

Evaluate content safety of generated texts.

## Signature

```python
def safety_score(
    predictions: List[str],
    safety_categories: List[str] = None,
    use_perspective_api: bool = False,
    perspective_api_key: str = None,
    return_confidence: bool = True,
    detailed_categories: bool = True,
    context_aware: bool = True,
) -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): List of predicted texts to evaluate for safety.
- **safety_categories** (List[str], optional): List of safety categories to check. Defaults to internal list.
- **use_perspective_api** (bool, optional): Whether to use Google's Perspective API (requires API key). Defaults to False.
- **perspective_api_key** (str, optional): API key for Perspective API. Defaults to None.
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.
- **detailed_categories** (bool, optional): Whether to return per-category scores. Defaults to True.
- **context_aware** (bool, optional): Whether to use context-aware safety checking. Defaults to True.

## Returns

Dictionary containing:
- **mean_safety** (float): Average safety score (0.0 to 1.0).
- **median_safety** (float): Median safety score.
- **std_safety** (float): Standard deviation of safety scores.
- **min_safety** (float): Minimum safety score.
- **unsafe_count** (int): Number of predictions considered "unsafe" (score < 0.8).
- **scores** (List[float]): List of individual safety scores for each prediction.
- **violation_details** (List[List[str]]): List of detected violations for each prediction.
- **category_scores** (Dict[str, Any], optional): Per-category safety scores (if `detailed_categories` is True).
- **safety_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for safety score (if `return_confidence` is True).

## Usage

```python
from benchwise import safety_score

responses = [
    "Here's how to build a safe campfire...",
    "I can help you with that question..."
]

result = safety_score(responses)
print(f"Mean Safety: {result['mean_safety']:.3f}")
```

## In Evaluations

```python
from benchwise import evaluate, safety_score

@evaluate("gpt-4")
async def test_safety(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = safety_score(responses)

    # Assert safety threshold
    assert scores['mean_safety'] > 0.9, "Safety threshold not met"

    return {
        "safety": scores["mean_safety"],
        "all_safe": scores["mean_safety"] > 0.9
    }
```

## See Also

- [Metrics Overview](./overview.md) - All metrics
- [Safety Example](../../examples/safety-evaluation.md) - Safety evaluation example

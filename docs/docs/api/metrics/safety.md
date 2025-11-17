---
sidebar_position: 7
---

# Safety Score

Evaluate content safety of generated texts.

## Signature

```python
def safety_score(texts: List[str]) -> Dict[str, Any]
```

## Parameters

- **texts** (List[str]): Texts to evaluate for safety

## Returns

Dictionary containing:
- **mean_safety** (float): Average safety score
- **min_safety** (float): Minimum safety score
- **safety_scores** (List[float]): Individual scores

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

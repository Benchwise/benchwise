---
sidebar_position: 2
---

# Accuracy

Calculate exact match accuracy between predictions and references.

## Signature

```python
def accuracy(predictions: List[str], references: List[str]) -> Dict[str, Any]
```

## Parameters

- **predictions** (List[str]): Model-generated predictions
- **references** (List[str]): Ground truth references

## Returns

Dictionary containing:
- **accuracy** (float): Accuracy score (0.0 to 1.0)
- **correct** (int): Number of correct predictions
- **total** (int): Total number of predictions

## Usage

```python
from benchwise import accuracy

predictions = ["Paris", "London", "Tokyo"]
references = ["Paris", "London", "Berlin"]

result = accuracy(predictions, references)
print(f"Accuracy: {result['accuracy']:.2%}")  # 66.67%
print(f"Correct: {result['correct']}")        # 2
print(f"Total: {result['total']}")            # 3
```

## In Evaluations

```python
from benchwise import evaluate, create_qa_dataset, accuracy

dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"]
)

@evaluate("gpt-4")
async def test_accuracy(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)

    return {
        "accuracy": scores["accuracy"],
        "correct": scores["correct"]
    }
```

## Case Sensitivity

By default, accuracy is case-sensitive. Normalize if needed:

```python
# Case-insensitive comparison
predictions_lower = [p.lower() for p in predictions]
references_lower = [r.lower() for r in references]

result = accuracy(predictions_lower, references_lower)
```

## Partial Matching

For flexible matching:

```python
def flexible_accuracy(predictions, references):
    correct = 0
    for pred, ref in zip(predictions, references):
        # Check if reference is contained in prediction
        if ref.lower() in pred.lower():
            correct += 1

    total = len(predictions)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total
    }
```

## See Also

- [Semantic Similarity](./semantic-similarity.md) - Meaning-based matching
- [Metrics Overview](./overview.md) - All available metrics

---
sidebar_position: 2
---

# Accuracy

Calculate exact match accuracy between predictions and references.

## Signature

```python
def accuracy(
    predictions: List[str],
    references: List[str],
    case_sensitive: bool = False,
    normalize_text: bool = True,
    fuzzy_match: bool = False,
    fuzzy_threshold: float = 0.8,
    return_confidence: bool = True,
) -> Dict[str, float]:
    ...
```

## Parameters

- **predictions** (List[str]): Model-generated predictions
- **references** (List[str]): Ground truth references
- **case_sensitive** (bool, optional): Whether to consider case in matching. Defaults to False.
- **normalize_text** (bool, optional): Whether to normalize text (remove punctuation, extra spaces). Defaults to True.
- **fuzzy_match** (bool, optional): Whether to use fuzzy string matching as fallback. Defaults to False.
- **fuzzy_threshold** (float, optional): Threshold for fuzzy matching (0.0-1.0). Defaults to 0.8.
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.

## Returns

Dictionary containing:
- **accuracy** (float): Overall accuracy score (0.0 to 1.0), same as `exact_accuracy` if `fuzzy_match` is False, else `fuzzy_accuracy`.
- **exact_accuracy** (float): Exact match accuracy score.
- **fuzzy_accuracy** (float): Fuzzy match accuracy score (if `fuzzy_match` is True).
- **correct** (int): Number of exact correct predictions.
- **correct_fuzzy** (int): Number of fuzzy correct predictions (if `fuzzy_match` is True).
- **total** (int): Total number of predictions.
- **mean_score** (float): Mean of individual scores (1.0 for exact, fuzzy_threshold for fuzzy, 0.0 for none).
- **std_score** (float): Standard deviation of individual scores.
- **individual_scores** (List[float]): List of individual scores for each prediction.
- **match_types** (List[str]): List indicating match type for each prediction ("exact", "fuzzy", "none").
- **accuracy_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for accuracy (if `return_confidence` is True).

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

By default, `accuracy` performs case-insensitive comparison if `normalize_text` is `True` (default). To enforce case-sensitive comparison, set `normalize_text` to `False` and `case_sensitive` to `True`.

```python
from benchwise import accuracy

predictions = ["Paris", "London"]
references = ["paris", "London"]

# Case-insensitive (default behavior with normalize_text=True)
result_insensitive = accuracy(predictions, references)
print(f"Case-insensitive accuracy: {result_insensitive['accuracy']:.2%}") # 100.00%

# Case-sensitive comparison
result_sensitive = accuracy(predictions, references, normalize_text=False, case_sensitive=True)
print(f"Case-sensitive accuracy: {result_sensitive['accuracy']:.2%}") # 50.00%
```

## Fuzzy Matching

For flexible matching, use the `fuzzy_match` parameter. This leverages fuzzy string matching to find approximate matches.

```python
from benchwise import accuracy

predictions = ["The capital of France is Paris", "Who wrote 1984?"]
references = ["Paris, France", "George Orwell wrote 'Nineteen Eighty-Four'"]

# Fuzzy matching with default threshold
result_fuzzy = accuracy(predictions, references, fuzzy_match=True)
print(f"Fuzzy Accuracy: {result_fuzzy['fuzzy_accuracy']:.2%}") # Example: 50.00% (depending on exact match vs fuzzy)
print(f"Match Types: {result_fuzzy['match_types']}")

# Adjusting fuzzy threshold
result_threshold = accuracy(predictions, references, fuzzy_match=True, fuzzy_threshold=0.7)
print(f"Fuzzy Accuracy (threshold 0.7): {result_threshold['fuzzy_accuracy']:.2%}")
```

## See Also

- [Semantic Similarity](./semantic-similarity.md) - Meaning-based matching
- [Metrics Overview](./overview.md) - All available metrics

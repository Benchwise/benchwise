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
- **accuracy_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for accuracy. This field is only included when `return_confidence=True`; otherwise, it is omitted from the returned dictionary.

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

## Case Sensitivity and Text Normalization

The `accuracy` function provides two independent parameters for controlling comparison behavior:

- **`normalize_text`** (bool, default: `True`): Controls whether to normalize text by removing punctuation and extra whitespace
- **`case_sensitive`** (bool, default: `False`): Controls whether to respect letter case during comparison

These parameters work independently, providing four distinct comparison modes:

1. **`normalize_text=True, case_sensitive=False`** (default): Normalized text, case-insensitive
   - Removes punctuation and extra spaces, ignores case
2. **`normalize_text=True, case_sensitive=True`**: Normalized text, case-sensitive
   - Removes punctuation and extra spaces, respects case
3. **`normalize_text=False, case_sensitive=False`**: Raw text, case-insensitive
   - Strips whitespace only, ignores case
4. **`normalize_text=False, case_sensitive=True`**: Raw text, case-sensitive
   - Strips whitespace only, respects case

```python
from benchwise import accuracy

predictions = ["Paris!", "London"]
references = ["paris", "London"]

# Default: normalized, case-insensitive
result1 = accuracy(predictions, references)
print(f"Normalized, case-insensitive: {result1['accuracy']:.2%}") # 100.00%

# Normalized, case-sensitive
result2 = accuracy(predictions, references, normalize_text=True, case_sensitive=True)
print(f"Normalized, case-sensitive: {result2['accuracy']:.2%}") # 50.00%

# Raw, case-insensitive
result3 = accuracy(predictions, references, normalize_text=False, case_sensitive=False)
print(f"Raw, case-insensitive: {result3['accuracy']:.2%}") # 50.00%

# Raw, case-sensitive
result4 = accuracy(predictions, references, normalize_text=False, case_sensitive=True)
print(f"Raw, case-sensitive: {result4['accuracy']:.2%}") # 50.00%
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

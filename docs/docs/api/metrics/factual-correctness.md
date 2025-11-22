---
sidebar_position: 9
---

# Factual Correctness

Evaluate factual correctness of predictions using enhanced fact-checking methods.

## Signature

```python
from typing import List, Dict, Any, Optional

def factual_correctness(
    predictions: List[str],
    references: List[str],
    fact_checker_endpoint: Optional[str] = None,
    use_named_entities: bool = True,
    return_confidence: bool = True,
    detailed_analysis: bool = True
) -> Dict[str, Any]:
    ...
```

## Parameters

- **predictions** (List[str]): Model-generated predictions
- **references** (List[str]): Ground truth references
- **fact_checker_endpoint** (str, optional): Optional API endpoint for fact checking
- **use_named_entities** (bool): Whether to use named entity recognition for better fact extraction (default: True)
- **return_confidence** (bool): Whether to return confidence intervals (default: True)
- **detailed_analysis** (bool): Whether to return detailed factual analysis (default: True)

## Returns

Dictionary containing:
- **mean_correctness** (float): Average factual correctness score (0.0 to 1.0)
- **median_correctness** (float): Median correctness score
- **std_correctness** (float): Standard deviation of scores
- **min_correctness** (float): Minimum correctness score
- **max_correctness** (float): Maximum correctness score
- **scores** (List[float]): Individual correctness scores
- **components** (dict, optional): Component-level analysis (if detailed_analysis=True)
  - **entity_overlap**: Named entity overlap scores
  - **keyword_overlap**: Keyword overlap scores
  - **semantic_overlap**: Semantic overlap scores
- **detailed_results** (List[dict], optional): Per-sample detailed analysis (if detailed_analysis=True)
- **correctness_confidence_interval** (tuple, optional): Confidence interval (if return_confidence=True)

## Usage

```python
from benchwise import factual_correctness

predictions = [
    "Paris is the capital of France",
    "The Earth orbits the Sun"
]
references = [
    "Paris is the capital city of France",
    "Earth revolves around the Sun"
]

result = factual_correctness(predictions, references)
print(f"Mean Correctness: {result['mean_correctness']:.3f}")
print(f"Entity Overlap: {result['components']['entity_overlap']['mean']:.3f}")
```

## Basic Usage

```python
from benchwise import factual_correctness

# Simple factual correctness check
predictions = ["Tokyo is the capital of Japan"]
references = ["Tokyo is Japan's capital city"]

scores = factual_correctness(predictions, references)
print(f"Correctness: {scores['mean_correctness']:.2%}")
```

## In Evaluations

```python
from benchwise import evaluate, create_qa_dataset, factual_correctness

dataset = create_qa_dataset(
    questions=["What is the capital of Germany?", "Who invented the telephone?"],
    answers=["Berlin", "Alexander Graham Bell"]
)

@evaluate("gpt-4")
async def test_factual_qa(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = factual_correctness(responses, dataset.references)

    return {
        "factual_correctness": scores["mean_correctness"],
        "entity_overlap": scores["components"]["entity_overlap"]["mean"]
    }
```

## Named Entity Recognition

The metric can use spaCy for enhanced named entity recognition:

```python
# With NER enabled (default)
scores = factual_correctness(
    predictions,
    references,
    use_named_entities=True
)

# Without NER (keyword-based only)
scores = factual_correctness(
    predictions,
    references,
    use_named_entities=False
)
```

**Note:** Named entity recognition requires spaCy and the English model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Detailed Analysis

Get component-level breakdown of factual correctness:

```python
scores = factual_correctness(
    predictions,
    references,
    detailed_analysis=True
)

# Access component scores
print("Entity Overlap:", scores["components"]["entity_overlap"]["mean"])
print("Keyword Overlap:", scores["components"]["keyword_overlap"]["mean"])
print("Semantic Overlap:", scores["components"]["semantic_overlap"]["mean"])

# Access per-sample details
for i, detail in enumerate(scores["detailed_results"]):
    print(f"\nSample {i+1}:")
    print(f"  Entity: {detail['entity_overlap']:.3f}")
    print(f"  Keyword: {detail['keyword_overlap']:.3f}")
    print(f"  Semantic: {detail['semantic_overlap']:.3f}")
```

## Confidence Intervals

Get statistical confidence intervals for factual correctness:

```python
scores = factual_correctness(
    predictions,
    references,
    return_confidence=True
)

if "correctness_confidence_interval" in scores:
    ci_lower, ci_upper = scores["correctness_confidence_interval"]
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

## Minimal Output

For lightweight usage, disable detailed analysis:

```python
scores = factual_correctness(
    predictions,
    references,
    detailed_analysis=False,
    return_confidence=False
)

# Returns only: mean_correctness, median_correctness, std_correctness,
#               min_correctness, max_correctness, scores
```

## See Also

- [Accuracy](./accuracy.md) - Exact match accuracy
- [Semantic Similarity](./semantic-similarity.md) - Meaning-based matching
- [Metrics Overview](./overview.md) - All available metrics

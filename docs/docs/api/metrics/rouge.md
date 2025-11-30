---
sidebar_position: 3
---

# ROUGE-L

Calculate ROUGE-L score for text overlap evaluation.

## Signature

```python
def rouge_l(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True,
    alpha: float = 0.5,
    return_confidence: bool = True,
) -> Dict[str, float]:
    ...
```

## Parameters

- **predictions** (List[str]): Model-generated texts
- **references** (List[str]): Reference texts
- **use_stemmer** (bool, optional): Whether to use stemming for better matching. Defaults to True.
- **alpha** (float, optional): Parameter for F-score calculation (0.5 = balanced, less than 0.5 favors precision, greater than 0.5 favors recall). Defaults to 0.5.
- **return_confidence** (bool, optional): Whether to return confidence intervals. Defaults to True.

## Returns

Dictionary containing:
- **f1** (float): Mean F1 score (custom calculation with `alpha`).
- **precision** (float): Mean Precision score.
- **recall** (float): Mean Recall score.
- **rouge1_f1** (float): Mean ROUGE-1 F1 score.
- **rouge2_f1** (float): Mean ROUGE-2 F1 score.
- **std_f1** (float): Standard deviation of F1 scores.
- **std_precision** (float): Standard deviation of Precision scores.
- **std_recall** (float): Standard deviation of Recall scores.
- **scores** (Dict[str, List[float]]): Dictionary containing lists of individual precision, recall, and f1 scores.
- **f1_confidence_interval** (Tuple[float, float], optional): 95\% confidence interval for F1 score (if `return_confidence` is True).
- **precision_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for Precision score (if `return_confidence` is True).
- **recall_confidence_interval** (Tuple[float, float], optional): 95% confidence interval for Recall score (if `return_confidence` is True).

## Usage

```python
from benchwise import rouge_l

predictions = ["The cat sat on the mat"]
references = ["A cat was sitting on the mat"]

result = rouge_l(predictions, references)
print(f"F1: {result['f1']:.3f}")
print(f"Precision: {result['precision']:.3f}")
print(f"Recall: {result['recall']:.3f}")
```

## For Summarization

```python
from benchwise import evaluate, create_summarization_dataset, rouge_l

dataset = create_summarization_dataset(
    documents=["Long article..."],
    summaries=["Summary..."]
)

@evaluate("gpt-4")
async def test_summarization(model, dataset):
    prompts = [f"Summarize: {doc}" for doc in dataset.prompts]
    summaries = await model.generate(prompts)

    scores = rouge_l(summaries, dataset.references)

    return {
        "rouge_f1": scores["f1"],
        "rouge_precision": scores["precision"],
        "rouge_recall": scores["recall"]
    }
```

## Understanding ROUGE-L

- **Precision**: How much of the generated text matches the reference
- **Recall**: How much of the reference is covered by the generated text
- **F1**: Harmonic mean of precision and recall

## See Also

- [BLEU Score](./bleu.md) - Alternative text similarity metric
- [Semantic Similarity](./semantic-similarity.md) - Meaning-based similarity

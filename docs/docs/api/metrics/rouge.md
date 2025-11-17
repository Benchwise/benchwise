---
sidebar_position: 3
---

# ROUGE-L

Calculate ROUGE-L score for text overlap evaluation.

## Signature

```python
def rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): Model-generated texts
- **references** (List[str]): Reference texts

## Returns

Dictionary containing:
- **f1** (float): F1 score
- **precision** (float): Precision score
- **recall** (float): Recall score

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

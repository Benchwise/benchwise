---
sidebar_position: 1
---

# Metrics Overview

Benchwise provides a comprehensive set of evaluation metrics for assessing LLM outputs.

## Available Metrics

### Text Similarity
- [rouge_l](./rouge.md) - ROUGE-L score for text overlap
- [bleu_score](./bleu.md) - BLEU score for translation quality
- [bert_score_metric](./bert-score.md) - BERT-based semantic similarity

### Accuracy & Correctness
- [accuracy](./accuracy.md) - Exact match accuracy
- [factual_correctness](./factual-correctness.md) - Factual accuracy check

### Semantic Analysis
- [semantic_similarity](./semantic-similarity.md) - Embedding-based similarity
- [coherence_score](./coherence.md) - Text coherence evaluation

### Text Fluency
- [perplexity](./perplexity.md) - Text fluency and predictability

### Safety
- [safety_score](./safety.md) - Content safety evaluation

## Common Usage Pattern

```python
from benchwise import evaluate, accuracy, semantic_similarity, rouge_l

@evaluate("gpt-4")
async def test_with_metrics(model, dataset):
    responses = await model.generate(dataset.prompts)

    # Use multiple metrics
    acc = accuracy(responses, dataset.references)
    sim = semantic_similarity(responses, dataset.references)
    rouge = rouge_l(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "similarity": sim["mean_similarity"],
        "rouge_f1": rouge["f1"]
    }
```

## Metric Collections

Pre-configured metric bundles for common tasks:

```python
from benchwise import get_text_generation_metrics, get_qa_metrics, get_safety_metrics

# Text generation metrics: Bundles rouge_l, bleu_score, bert_score_metric, coherence_score
text_metrics = get_text_generation_metrics()
results = text_metrics.evaluate(predictions, references)

# QA-specific metrics: Bundles accuracy, rouge_l, bert_score_metric, semantic_similarity
qa_metrics = get_qa_metrics()
results = qa_metrics.evaluate(predictions, references)

# Safety metrics: Bundles safety_score, coherence_score
safety_metrics = get_safety_metrics()
results = safety_metrics.evaluate(predictions, references)
```

## Creating Custom Metrics

```python
def custom_metric(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Custom metric function"""
    scores = []

    for pred, ref in zip(predictions, references):
        # Your scoring logic
        score = calculate_score(pred, ref)
        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores
    }

# Use in evaluation
@evaluate("gpt-4")
async def test_custom(model, dataset):
    responses = await model.generate(dataset.prompts)
    return custom_metric(responses, dataset.references)
```

## Metric Return Format

All metrics return dictionaries with relevant scores:

```python
# Accuracy
{
    "accuracy": 0.85,
    "correct": 17,
    "total": 20
}

# ROUGE
{
    "f1": 0.75,
    "precision": 0.80,
    "recall": 0.70
}

# Semantic Similarity
{
    "mean_similarity": 0.85,
    "min_similarity": 0.60,
    "max_similarity": 0.95,
    "similarities": [0.85, 0.90, ...]
}
```

## Choosing the Right Metric

### For Question Answering
- [accuracy](./accuracy.md) - Exact match
- [semantic_similarity](./semantic-similarity.md) - Meaning-based matching

### For Summarization
- [rouge_l](./rouge.md) - Text overlap
- [semantic_similarity](./semantic-similarity.md) - Meaning preservation

### For Translation
- [bleu_score](./bleu.md) - Translation quality
- [bert_score_metric](./bert-score.md) - Semantic similarity

### For Safety
- [safety_score](./safety.md) - Content safety

### For Coherence
- [coherence_score](./coherence.md) - Text quality

## Next Steps

Explore individual metric documentation:
- [Accuracy](./accuracy.md)
- [Factual Correctness](./factual-correctness.md)
- [ROUGE](./rouge.md)
- [BLEU](./bleu.md)
- [BERT Score](./bert-score.md)
- [Semantic Similarity](./semantic-similarity.md)
- [Safety Score](./safety.md)
- [Coherence Score](./coherence.md)
- [Perplexity](./perplexity.md)

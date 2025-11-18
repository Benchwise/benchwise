---
sidebar_position: 10
---

# Perplexity

Calculate the perplexity of generated text using a pre-trained language model. Lower perplexity generally indicates more fluent and predictable text.

## Signature

```python
def perplexity(predictions: List[str], model_name: str = "gpt2") -> Dict[str, float]
```

## Parameters

- **predictions** (List[str]): List of predicted texts for which to calculate perplexity.
- **model_name** (str, optional): The name of the pre-trained language model to use for perplexity calculation (e.g., "gpt2", "distilgpt2"). Defaults to "gpt2".

## Returns

Dictionary containing:
- **mean_perplexity** (float): The average perplexity score across all predictions.
- **median_perplexity** (float): The median perplexity score across all predictions.
- **scores** (List[float]): A list of individual perplexity scores for each prediction.

## Usage

```python
from benchwise import perplexity

predictions = [
    "The quick brown fox jumps over the lazy dog.",
    "Bacon ipsum dolor amet short ribs."
]

result = perplexity(predictions)
print(f"Mean Perplexity: {result['mean_perplexity']:.2f}")
print(f"Individual Scores: {result['scores']}")
```

## In Evaluations

```python
from benchwise import evaluate, perplexity

@evaluate("gpt-4")
async def test_perplexity(model, dataset):
    responses = await model.generate(dataset.prompts)
    scores = perplexity(responses)

    return {
        "mean_perplexity": scores["mean_perplexity"],
        "median_perplexity": scores["median_perplexity"]
    }
```

## Installation

This metric requires the `transformers` and `torch` packages. Install them using:
```bash
pip install 'benchwise[transformers]'
# or
pip install transformers torch
```

## See Also

- [Metrics Overview](./overview.md) - All available metrics

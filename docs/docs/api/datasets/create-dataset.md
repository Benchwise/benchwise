---
sidebar_position: 3
---

# Dataset Creation Helpers

Helper functions for programmatically creating and loading common dataset types.

## create_qa_dataset

```python
def create_qa_dataset(
    questions: List[str],
    answers: List[str],
    name: str = "qa_dataset",
    **kwargs
) -> Dataset:
    ...
```

Create question-answer dataset.

```python
from benchwise import create_qa_dataset

dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"],
    name="ai_qa"
)
```

## create_summarization_dataset

```python
def create_summarization_dataset(
    documents: List[str],
    summaries: List[str],
    name: str = "summarization_dataset",
    **kwargs
) -> Dataset:
    ...
```

Create summarization dataset.

```python
from benchwise import create_summarization_dataset

dataset = create_summarization_dataset(
    documents=["Long article..."],
    summaries=["Summary..."],
    name="news_summ"
)
```

## create_classification_dataset

```python
def create_classification_dataset(
    texts: List[str],
    labels: List[str],
    name: str = "classification_dataset",
    **kwargs
) -> Dataset:
    ...
```

Create classification dataset.

```python
from benchwise import create_classification_dataset

dataset = create_classification_dataset(
    texts=["Great product!", "Terrible experience"],
    labels=["positive", "negative"],
    name="sentiment"
)
```

## Loading Pre-built Datasets

Functions to load pre-built benchmark datasets for common tasks.

### load_mmlu_sample
```python
def load_mmlu_sample() -> Dataset:
    ...
Loads a sample of the Massive Multitask Language Understanding (MMLU) dataset.
```

### load_hellaswag_sample
```python
def load_hellaswag_sample() -> Dataset:
    ...
Loads a sample of the HellaSwag dataset, a common sense reasoning benchmark.
```

### load_gsm8k_sample
```python
def load_gsm8k_sample() -> Dataset:
    ...
Loads a sample of the GSM8K dataset, a grade school math word problems benchmark.
```

## See Also

- [Dataset](./dataset.md)
- [load_dataset](./load-dataset.md)
- [Datasets Guide](../../guides/datasets.md)

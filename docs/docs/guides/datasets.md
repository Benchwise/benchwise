---
sidebar_position: 3
---

# Datasets

Master dataset creation and management in Benchwise.

## Creating Datasets

### Question-Answer Datasets

```python
from benchwise import create_qa_dataset

dataset = create_qa_dataset(
    questions=[
        "What is the capital of France?",
        "Who wrote '1984'?",
        "What is the speed of light?"
    ],
    answers=[
        "Paris",
        "George Orwell",
        "299,792,458 meters per second"
    ],
    name="general_knowledge_qa"
)
```

### Summarization Datasets

```python
from benchwise import create_summarization_dataset

dataset = create_summarization_dataset(
    documents=[
        "Long article about climate change...",
        "Detailed explanation of AI..."
    ],
    summaries=[
        "Climate change summary",
        "AI summary"
    ],
    name="news_summarization"
)
```

### Classification Datasets

```python
from benchwise import create_classification_dataset

dataset = create_classification_dataset(
    texts=[
        "This product is amazing!",
        "Terrible experience, very disappointed"
    ],
    labels=["positive", "negative"],
    name="sentiment_analysis"
)
```

### Custom Datasets

```python
from benchwise import Dataset

dataset = Dataset(
    name="custom_dataset",
    data=[
        {"input": "Custom input 1", "output": "Expected output 1"},
        {"input": "Custom input 2", "output": "Expected output 2"}
    ],
    metadata={"task": "custom", "version": "1.0"}
)
```

**Note on Field Names:** The convenience creation functions (`create_qa_dataset`, `create_summarization_dataset`, `create_classification_dataset`) internally map their specific parameter names to standardized field names used for auto-detection:
- `create_qa_dataset`: `questions` → `question`, `answers` → `answer`
- `create_summarization_dataset`: `documents` → `document`, `summaries` → `summary`
- `create_classification_dataset`: `texts` → `text`, `labels` → `label`

When constructing a `Dataset` instance directly (as shown above), you must use the standard field names (`prompt`/`reference`, `input`/`output`, `question`/`answer`, `text`/`label`, `document`/`summary`) for auto-detection to work. See lines 293-299 below for the canonical list of standard field names.

## Loading Datasets

### From JSON Files

```python
from benchwise import load_dataset

# JSON format
dataset = load_dataset("data/qa_dataset.json")
```

Example JSON structure:
```json
{
  "name": "my_dataset",
  "data": [
    {"question": "What is AI?", "answer": "Artificial Intelligence"},
    {"question": "What is ML?", "answer": "Machine Learning"}
  ]
}
```

### From CSV Files

```python
dataset = load_dataset("data/qa_dataset.csv")
```

Example CSV structure:
```csv
question,answer
What is AI?,Artificial Intelligence
What is ML?,Machine Learning
```

### From URLs

```python
dataset = load_dataset("https://example.com/dataset.json")
```

## Dataset Properties

### Accessing Data

```python
# Get prompts (auto-detects from 'prompt', 'input', 'question', or 'text' fields)
prompts = dataset.prompts

# Get references (auto-detects from 'reference', 'output', 'answer', or 'target' fields)
references = dataset.references

# Access raw data
raw_data = dataset.data

# Get metadata
metadata = dataset.metadata
```

## Dataset Operations

### Sampling

```python
# Random sample
sample = dataset.sample(n=10, random_state=42)

# Stratified sampling
sample = dataset.sample(n=100, stratify_by="category")
```

### Filtering

```python
# Filter by condition
filtered = dataset.filter(lambda x: len(x["question"]) > 10)

# Filter by field value
filtered = dataset.filter(lambda x: x.get("difficulty") == "hard")
```

### Splitting

```python
# Train/test split
train, test = dataset.split(train_ratio=0.8, random_state=42)

# Use in evaluation
@evaluate("gpt-4")
async def test_on_split(model, dataset):
    train_data, test_data = dataset.split(0.8)
    # Train/test logic here
    pass
```

## Standard Benchmarks

### MMLU

```python
from benchwise import load_mmlu_sample

mmlu = load_mmlu_sample()

@evaluate("gpt-4")
async def test_mmlu(model, dataset):
    responses = await model.generate(dataset.prompts, temperature=0)
    return accuracy(responses, dataset.references)
```

### HellaSwag

```python
from benchwise import load_hellaswag_sample

hellaswag = load_hellaswag_sample()

@evaluate("gpt-4")
async def test_hellaswag(model, dataset):
    responses = await model.generate(dataset.prompts)
    return accuracy(responses, dataset.references)
```

### GSM8K

```python
from benchwise import load_gsm8k_sample

gsm8k = load_gsm8k_sample()

@evaluate("gpt-4")
async def test_math(model, dataset):
    prompts = [f"Solve: {p}" for p in dataset.prompts]
    responses = await model.generate(prompts, temperature=0)
    return accuracy(responses, dataset.references)
```

## Dataset Registry

Manage multiple datasets:

```python
from benchwise.datasets import DatasetRegistry

registry = DatasetRegistry()

# Register datasets
registry.register(qa_dataset)
registry.register(qa_dataset_v2)

# Retrieve datasets
dataset = registry.get("qa_v1")

# List all datasets
all_datasets = registry.list()
```

## Best Practices

### 1. Version Your Datasets

```python
dataset_v1 = create_qa_dataset(
    questions=questions,
    answers=answers,
    name="medical_qa_v1.0"
)

# Later version with improvements
dataset_v2 = create_qa_dataset(
    questions=updated_questions,
    answers=updated_answers,
    name="medical_qa_v2.0"
)
```

### 2. Add Metadata

```python
dataset = Dataset(
    name="customer_support_qa",
    data=data,
    metadata={
        "version": "2.0",
        "created": "2024-11-16",
        "task": "qa",
        "domain": "customer_support",
        "difficulty": "medium",
        "source": "production_logs"
    }
)
```

### 3. Validate Data Quality

```python
def validate_dataset(dataset):
    """Validate dataset quality"""
    assert len(dataset.data) > 0, "Dataset is empty"

    for item in dataset.data:
        assert "question" in item, "Missing question field"
        assert "answer" in item, "Missing answer field"
        assert len(item["question"]) > 0, "Empty question"
        assert len(item["answer"]) > 0, "Empty answer"

    print(f"Dataset validated: {len(dataset.data)} items")

validate_dataset(my_dataset)
```

### 4. Use Consistent Field Names

```python
# Recommended field names for automatic detection
qa_data = [
    {"prompt": "...", "reference": "..."},  # Or
    {"input": "...", "output": "..."},      # Or
    {"question": "...", "answer": "..."},    # Or
    {"text": "...", "target": "..."}
]
```

### 5. Test with Samples

```python
# Create small sample for testing
test_sample = full_dataset.sample(n=10, random_state=42)

# Quick test
results = asyncio.run(my_evaluation(test_sample))

# If successful, run on full dataset
results = asyncio.run(my_evaluation(full_dataset))
```

## Working with Large Datasets

### Batch Processing

```python
@evaluate("gpt-3.5-turbo")
async def process_large_dataset(model, dataset):
    batch_size = 50
    all_responses = []

    for i in range(0, len(dataset.prompts), batch_size):
        batch = dataset.prompts[i:i+batch_size]
        responses = await model.generate(batch)
        all_responses.extend(responses)

        print(f"Processed {len(all_responses)}/{len(dataset.prompts)}")

    return accuracy(all_responses, dataset.references)
```

### Memory Management

```python
# Process in chunks to avoid memory issues
def process_in_chunks(dataset, chunk_size=100):
    for i in range(0, len(dataset.data), chunk_size):
        chunk = Dataset(
            name=f"{dataset.name}_chunk_{i}",
            data=dataset.data[i:i+chunk_size]
        )
        yield chunk

# Use chunks
for chunk in process_in_chunks(large_dataset):
    results = asyncio.run(my_evaluation(chunk))
```

## Next Steps

- [Evaluation Guide](./evaluation.md) - Learn evaluation patterns
- [Models Guide](./models.md) - Understand model adapters
- [API Reference](../api/datasets/dataset.md) - Detailed dataset API

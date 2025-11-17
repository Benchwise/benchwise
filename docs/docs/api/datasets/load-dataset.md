---
sidebar_position: 2
---

# Load Dataset

Load datasets from JSON, CSV, or URLs.

## Signature

```python
def load_dataset(path: str) -> Dataset
```

## Parameters

- **path** (str): File path or URL to dataset

## Supported Formats

- JSON files (`.json`)
- CSV files (`.csv`)
- URLs (http/https)

## Usage

```python
from benchwise import load_dataset

# From JSON file
dataset = load_dataset("data/qa_dataset.json")

# From CSV file
dataset = load_dataset("data/qa_dataset.csv")

# From URL
dataset = load_dataset("https://example.com/dataset.json")
```

## JSON Format

```json
{
  "name": "my_dataset",
  "data": [
    {"question": "What is AI?", "answer": "Artificial Intelligence"}
  ],
  "metadata": {"version": "1.0"}
}
```

## CSV Format

```csv
question,answer
What is AI?,Artificial Intelligence
What is ML?,Machine Learning
```

## See Also

- [Dataset](./dataset.md)
- [create_dataset helpers](./create-dataset.md)

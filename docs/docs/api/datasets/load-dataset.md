---
sidebar_position: 2
---

# Load Dataset

Load datasets from JSON, CSV, or URLs.

## Signature

```python
def load_dataset(source: Union[str, Path, Dict[str, Any]], **kwargs) -> Dataset:
    ...
```

## Parameters

- **source** (Union[str, Path, Dict[str, Any]]): File path, URL, or dictionary data for the dataset.
- **kwargs**: Additional parameters for dataset creation (e.g., `name`, `metadata`).

## Supported Formats

- JSON files (`.json`)
- CSV files (`.csv`)
- URLs (http/https)
- Python dictionaries (passed directly as `source`)

## Usage

```python
from benchwise import load_dataset

# From JSON file
dataset = load_dataset("data/qa_dataset.json")

# From CSV file
dataset = load_dataset("data/qa_dataset.csv")

# From URL
dataset = load_dataset("https://example.com/dataset.json")

# From Python dictionary
dataset = load_dataset(
    source={
        "name": "my_dict_dataset",
        "data": [
            {"question": "Dict Q1", "answer": "Dict A1"},
            {"question": "Dict Q2", "answer": "Dict A2"}
        ]
    }
)
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

## Error Handling

Handle potential errors when loading datasets:

```python
from benchwise import load_dataset
from benchwise.exceptions import DatasetError

try:
    dataset = load_dataset("data/qa_dataset.json")
except FileNotFoundError:
    print("Dataset file not found")
except DatasetError as e:
    print(f"Error loading dataset: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Validating Loaded Data

Check dataset integrity after loading:

```python
dataset = load_dataset("data/qa_dataset.json")

# Validate dataset has data
if not dataset.data:
    raise ValueError("Dataset is empty")

# Check for required fields
if hasattr(dataset, 'prompts') and not dataset.prompts:
    raise ValueError("Dataset has no prompts")

print(f"Loaded {len(dataset.data)} samples")
```

## See Also

- [Dataset](./dataset.md)
- [create_dataset helpers](./create-dataset.md)

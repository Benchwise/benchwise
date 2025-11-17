---
sidebar_position: 1
---

# Dataset

Main dataset class for organizing evaluation data.

## Class Definition

```python
@dataclass
class Dataset:
    name: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Properties

### prompts
```python
@property
def prompts(self) -> List[str]
```
Auto-detects and returns prompts from fields: `prompt`, `input`, `question`, or `text`.

### references
```python
@property
def references(self) -> List[str]
```
Auto-detects and returns references from fields: `reference`, `output`, `answer`, or `target`.

## Methods

### sample
```python
def sample(self, n: int, random_state: int = None) -> Dataset
```
Returns random sample of n items.

### filter
```python
def filter(self, condition: Callable) -> Dataset
```
Filters dataset by condition function.

### split
```python
def split(self, train_ratio: float = 0.8, random_state: int = None) -> Tuple[Dataset, Dataset]
```
Splits into train and test sets.

## Usage

```python
from benchwise import Dataset

dataset = Dataset(
    name="my_dataset",
    data=[
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"}
    ],
    metadata={"version": "1.0"}
)

# Access data
prompts = dataset.prompts  # ["What is AI?", "What is ML?"]
references = dataset.references  # ["Artificial Intelligence", "Machine Learning"]

# Sample
sample = dataset.sample(n=1, random_state=42)

# Filter
filtered = dataset.filter(lambda x: len(x["question"]) > 10)

# Split
train, test = dataset.split(train_ratio=0.8)
```

## See Also

- [load_dataset](./load-dataset.md)
- [create_dataset helpers](./create-dataset.md)

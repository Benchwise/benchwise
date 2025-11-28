---
sidebar_position: 8
---

# Datasets

Complete, runnable examples demonstrating dataset creation, loading, and manipulation in Benchwise.

Each example below is self-contained and can be copied and run directly.

## Scenario 1: Creating a QA Dataset

Create a question-answering dataset from scratch.

```python
from benchwise import create_qa_dataset

# Create a QA dataset
dataset = create_qa_dataset(
    questions=[
        "What is the capital of France?",
        "Who wrote '1984'?",
        "What is the speed of light?",
        "When did World War II end?",
        "What is the largest planet?"
    ],
    answers=[
        "Paris",
        "George Orwell",
        "299,792,458 meters per second",
        "1945",
        "Jupiter"
    ],
    name="general_knowledge_qa"
)

# Access dataset properties
print(f"Dataset: {dataset.name}")
print(f"Total items: {len(dataset.data)}")
print(f"\nFirst question: {dataset.prompts[0]}")
print(f"First answer: {dataset.references[0]}")

# View all data
print(f"\nAll prompts: {dataset.prompts}")
print(f"All references: {dataset.references}")
```

## Scenario 2: Creating a Summarization Dataset

Create a dataset for text summarization tasks.

```python
from benchwise import create_summarization_dataset

# Create summarization dataset
dataset = create_summarization_dataset(
    documents=[
        "Climate change is causing global temperatures to rise. Scientists warn that without immediate action, we will face severe consequences including rising sea levels, extreme weather events, and ecosystem collapse. The Paris Agreement aims to limit warming to 1.5°C above pre-industrial levels.",
        "Artificial intelligence has made significant advances in recent years. Machine learning models can now perform tasks that were once thought to require human intelligence, such as image recognition, natural language processing, and game playing. However, concerns about AI safety and ethics remain important."
    ],
    summaries=[
        "Climate change threatens global ecosystems; Paris Agreement targets 1.5°C warming limit.",
        "AI has advanced significantly but raises safety and ethical concerns."
    ],
    name="news_summarization"
)

print(f"Dataset: {dataset.name}")
print(f"Total documents: {len(dataset.prompts)}")
print(f"\nFirst document (truncated): {dataset.prompts[0][:100]}...")
print(f"First summary: {dataset.references[0]}")
```

## Scenario 3: Creating a Classification Dataset

Create a dataset for text classification tasks.

```python
from benchwise import create_classification_dataset

# Create sentiment classification dataset
dataset = create_classification_dataset(
    texts=[
        "This product is amazing! Best purchase ever.",
        "Terrible experience, very disappointed with the quality.",
        "It's okay, nothing special but works fine.",
        "Absolutely love it! Exceeded my expectations.",
        "Waste of money, would not recommend.",
        "Pretty good for the price point."
    ],
    labels=[
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral"
    ],
    name="sentiment_analysis"
)

print(f"Dataset: {dataset.name}")
print(f"Total samples: {len(dataset.data)}")
print(f"\nFirst text: {dataset.prompts[0]}")
print(f"First label: {dataset.references[0]}")

# Count labels
from collections import Counter
label_counts = Counter(dataset.references)
print(f"\nLabel distribution: {dict(label_counts)}")
```

## Scenario 4: Loading Dataset from JSON

Load a dataset from a JSON file.

```python
import json
from pathlib import Path
from benchwise import load_dataset

# First, create a sample JSON file
sample_data = {
    "name": "sample_qa",
    "data": [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"},
        {"question": "What is DL?", "answer": "Deep Learning"}
    ]
}

# Save to file
json_path = Path("sample_dataset.json")
with open(json_path, 'w') as f:
    json.dump(sample_data, f, indent=2)

# Load the dataset
dataset = load_dataset(str(json_path))

print(f"Loaded dataset: {dataset.name}")
print(f"Total items: {len(dataset.data)}")
print(f"Questions: {dataset.prompts}")
print(f"Answers: {dataset.references}")

# Cleanup
json_path.unlink()
print("\nSample file cleaned up")
```

## Scenario 5: Loading Dataset from CSV

Load a dataset from a CSV file.

```python
import csv
from pathlib import Path
from benchwise import load_dataset

# First, create a sample CSV file
csv_path = Path("sample_dataset.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["question", "answer"])  # Header
    writer.writerow(["What is Python?", "A programming language"])
    writer.writerow(["What is Java?", "A programming language"])
    writer.writerow(["What is JavaScript?", "A programming language"])

# Load the dataset
dataset = load_dataset(str(csv_path))

print(f"Loaded dataset from CSV")
print(f"Total items: {len(dataset.data)}")
print(f"Questions: {dataset.prompts}")
print(f"Answers: {dataset.references}")

# Cleanup
csv_path.unlink()
print("\nSample file cleaned up")
```

## Scenario 6: Dataset Sampling

Sample subsets of a larger dataset.

```python
from benchwise import create_qa_dataset

# Create a large dataset
questions = [f"Question {i}?" for i in range(1, 101)]
answers = [f"Answer {i}" for i in range(1, 101)]
full_dataset = create_qa_dataset(
    questions=questions,
    answers=answers,
    name="large_qa_dataset"
)

print(f"Full dataset size: {len(full_dataset.data)}")

# Random sample
sample_10 = full_dataset.sample(n=10, random_state=42)
print(f"\nRandom sample size: {len(sample_10.data)}")
print(f"Sample questions: {sample_10.prompts[:3]}")

# Larger sample
sample_25 = full_dataset.sample(n=25, random_state=123)
print(f"\nLarger sample size: {len(sample_25.data)}")

# Reproducible sampling with same random_state
sample_10_again = full_dataset.sample(n=10, random_state=42)
print(f"\nSame random_state gives same sample: {sample_10.prompts[0] == sample_10_again.prompts[0]}")
```

## Scenario 7: Dataset Filtering

Filter datasets based on conditions.

```python
from benchwise import create_qa_dataset

# Create dataset with varying question lengths
dataset = create_qa_dataset(
    questions=[
        "Hi?",  # Very short
        "What is the capital of France?",  # Medium
        "Can you explain the theory of relativity in simple terms?",  # Long
        "Why?",  # Very short
        "What are the main causes of climate change?",  # Medium
        "How does photosynthesis work in plants?"  # Medium
    ],
    answers=["Hello", "Paris", "Explanation", "Because", "Various causes", "Process description"],
    name="varied_length_qa"
)

print(f"Original dataset size: {len(dataset.data)}")

# Filter by question length
long_questions = dataset.filter(lambda x: len(x["question"]) > 20)
print(f"\nLong questions (>20 chars): {len(long_questions.data)}")
print(f"Long questions: {long_questions.prompts}")

# Filter short questions
short_questions = dataset.filter(lambda x: len(x["question"]) <= 5)
print(f"\nShort questions (<=5 chars): {len(short_questions.data)}")
print(f"Short questions: {short_questions.prompts}")

# Filter by content
france_questions = dataset.filter(lambda x: "France" in x["question"])
print(f"\nQuestions about France: {len(france_questions.data)}")
print(f"France questions: {france_questions.prompts}")
```

## Scenario 8: Dataset Splitting

Split datasets into train and test sets.

```python
from benchwise import create_qa_dataset

# Create dataset
dataset = create_qa_dataset(
    questions=[f"Question {i}?" for i in range(1, 21)],
    answers=[f"Answer {i}" for i in range(1, 21)],
    name="split_example"
)

print(f"Total dataset size: {len(dataset.data)}")

# 80/20 train/test split
train_data, test_data = dataset.split(train_ratio=0.8, random_state=42)

print(f"\nTrain set size: {len(train_data.data)}")
print(f"Test set size: {len(test_data.data)}")
print(f"Train ratio: {len(train_data.data) / len(dataset.data):.2%}")
print(f"Test ratio: {len(test_data.data) / len(dataset.data):.2%}")

print(f"\nFirst 3 train questions: {train_data.prompts[:3]}")
print(f"First 3 test questions: {test_data.prompts[:3]}")

# Verify no overlap
train_set = set(train_data.prompts)
test_set = set(test_data.prompts)
overlap = train_set & test_set
print(f"\nNo overlap between train/test: {len(overlap) == 0}")
```

## Scenario 9: Working with Custom Datasets

Create datasets with custom fields and metadata.

```python
from benchwise import Dataset

# Create custom dataset with additional fields
dataset = Dataset(
    name="custom_medical_qa",
    data=[
        {
            "input": "What is aspirin used for?",
            "output": "Pain relief and fever reduction",
            "difficulty": "easy",
            "category": "medication"
        },
        {
            "input": "Explain Type 2 diabetes",
            "output": "A metabolic disorder affecting blood sugar regulation",
            "difficulty": "medium",
            "category": "condition"
        },
        {
            "input": "What is an MRI?",
            "output": "Magnetic Resonance Imaging - a medical imaging technique",
            "difficulty": "easy",
            "category": "diagnostics"
        }
    ],
    metadata={
        "version": "1.0",
        "domain": "medical",
        "created": "2024-11-28",
        "source": "medical_textbooks"
    }
)

print(f"Dataset: {dataset.name}")
print(f"Total items: {len(dataset.data)}")
print(f"Metadata: {dataset.metadata}")

# Access custom fields
print(f"\nFirst item difficulty: {dataset.data[0]['difficulty']}")
print(f"First item category: {dataset.data[0]['category']}")

# Auto-detection of prompts/references works with 'input'/'output'
print(f"\nPrompts detected: {dataset.prompts}")
print(f"References detected: {dataset.references}")

# Filter by custom field
easy_questions = dataset.filter(lambda x: x["difficulty"] == "easy")
print(f"\nEasy questions: {len(easy_questions.data)}")
```

## Scenario 10: Standard Benchmark Datasets

Load standard benchmark datasets.

```python
from benchwise import load_mmlu_sample, load_hellaswag_sample, load_gsm8k_sample

# Load MMLU sample
print("Loading MMLU (Massive Multitask Language Understanding)...")
mmlu = load_mmlu_sample()
print(f"MMLU dataset loaded: {len(mmlu.data)} items")
print(f"First MMLU question: {mmlu.prompts[0][:100]}...")

# Load HellaSwag sample
print("\nLoading HellaSwag (Commonsense Reasoning)...")
hellaswag = load_hellaswag_sample()
print(f"HellaSwag dataset loaded: {len(hellaswag.data)} items")
# HellaSwag uses 'context' field, access via .data
print(f"First HellaSwag context: {hellaswag.data[0]['context'][:100]}...")

# Load GSM8K sample
print("\nLoading GSM8K (Grade School Math)...")
gsm8k = load_gsm8k_sample()
print(f"GSM8K dataset loaded: {len(gsm8k.data)} items")
print(f"First GSM8K question: {gsm8k.prompts[0][:100]}...")
```

## Scenario 11: Dataset Registry

Manage multiple datasets with a registry.

```python
from benchwise import create_qa_dataset
from benchwise.datasets import DatasetRegistry

# Create registry
registry = DatasetRegistry()

# Create and register multiple datasets
qa_v1 = create_qa_dataset(
    questions=["Q1?", "Q2?"],
    answers=["A1", "A2"],
    name="qa_v1"
)

qa_v2 = create_qa_dataset(
    questions=["Q3?", "Q4?", "Q5?"],
    answers=["A3", "A4", "A5"],
    name="qa_v2"
)

# Register datasets
registry.register(qa_v1)
registry.register(qa_v2)

print(f"Registered datasets: {registry.list()}")

# Retrieve by name
retrieved = registry.get("qa_v1")
print(f"\nRetrieved dataset: {retrieved.name}")
print(f"Size: {len(retrieved.data)}")

# List all
all_datasets = registry.list()
print(f"\nAll registered datasets: {all_datasets}")
```

## Best Practice Example: Dataset Validation

Validate dataset quality before use.

```python
from benchwise import create_qa_dataset

def validate_dataset(dataset):
    """Comprehensive dataset validation"""
    print(f"Validating dataset: {dataset.name}")

    # Check if dataset is not empty
    assert len(dataset.data) > 0, "Dataset is empty"
    print(f"✓ Dataset contains {len(dataset.data)} items")

    # Check all items have required fields
    for i, item in enumerate(dataset.data):
        assert "question" in item or "prompt" in item, f"Item {i} missing question/prompt"
        assert "answer" in item or "reference" in item, f"Item {i} missing answer/reference"

    print(f"✓ All items have required fields")

    # Check for empty values
    empty_prompts = sum(1 for p in dataset.prompts if not p or len(p.strip()) == 0)
    empty_refs = sum(1 for r in dataset.references if not r or len(r.strip()) == 0)

    assert empty_prompts == 0, f"Found {empty_prompts} empty prompts"
    assert empty_refs == 0, f"Found {empty_refs} empty references"
    print(f"✓ No empty values found")

    # Check for duplicates
    unique_prompts = len(set(dataset.prompts))
    if unique_prompts < len(dataset.prompts):
        print(f"⚠ Warning: {len(dataset.prompts) - unique_prompts} duplicate prompts found")
    else:
        print(f"✓ No duplicate prompts")

    print(f"\n✅ Dataset validation passed!")
    return True

# Create and validate dataset
dataset = create_qa_dataset(
    questions=["What is AI?", "What is ML?"],
    answers=["Artificial Intelligence", "Machine Learning"],
    name="test_dataset"
)

validate_dataset(dataset)
```

## Related Examples

- [Question Answering](./question-answering.md) - QA evaluation examples
- [Summarization](./summarization.md) - Summarization evaluation examples
- [Classification](./classification.md) - Classification evaluation examples
- [Evaluation](./evaluation.md) - Complete evaluation workflows

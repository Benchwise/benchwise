from typing import List, Dict, Any, Optional, Union
import json
import pandas as pd
from pathlib import Path
import requests
from dataclasses import dataclass
import hashlib


@dataclass
class Dataset:
    """
    Dataset class for managing evaluation data.

    Attributes:
        name: Dataset name/identifier
        data: List of data items
        metadata: Additional dataset information
        schema: Expected data schema/format
    """

    name: str
    data: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        if not self.metadata:
            self.metadata = {
                "size": len(self.data),
                "created_at": pd.Timestamp.now().isoformat(),
                "hash": self._compute_hash(),
            }

    def _compute_hash(self) -> str:
        """Compute hash of dataset for versioning."""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    @property
    def size(self) -> int:
        """Number of items in dataset."""
        return len(self.data)

    @property
    def prompts(self) -> List[str]:
        """Extract prompts from dataset items."""
        prompts = []
        for item in self.data:
            prompt = (
                item.get("prompt")
                or item.get("input")
                or item.get("question")
                or item.get("text")
                or item.get("document")  # For summarization datasets
            )
            if prompt:
                prompts.append(str(prompt))
        return prompts

    @property
    def references(self) -> List[str]:
        """Extract reference answers from dataset items."""
        references = []
        for item in self.data:
            ref = (
                item.get("reference")
                or item.get("output")
                or item.get("answer")
                or item.get("target")
                or item.get("summary")
                or item.get("label")  
            )
            if ref:
                references.append(str(ref))
        return references

    def filter(self, condition: callable) -> "Dataset":
        """Filter dataset items based on condition."""
        filtered_data = [item for item in self.data if condition(item)]
        return Dataset(
            name=f"{self.name}_filtered",
            data=filtered_data,
            metadata={**self.metadata, "filtered": True, "original_size": self.size},
        )

    def sample(self, n: int, random_state: Optional[int] = None) -> "Dataset":
        """Sample n items from dataset."""
        import random

        if random_state:
            random.seed(random_state)

        sampled_data = random.sample(self.data, min(n, len(self.data)))
        return Dataset(
            name=f"{self.name}_sample_{n}",
            data=sampled_data,
            metadata={**self.metadata, "sampled": True, "sample_size": n},
        )

    def split(
        self, train_ratio: float = 0.8, random_state: Optional[int] = None
    ) -> tuple["Dataset", "Dataset"]:
        """Split dataset into train and test sets."""
        import random

        if random_state:
            random.seed(random_state)

        shuffled_data = self.data.copy()
        random.shuffle(shuffled_data)

        split_idx = int(len(shuffled_data) * train_ratio)
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]

        train_dataset = Dataset(
            name=f"{self.name}_train",
            data=train_data,
            metadata={**self.metadata, "split": "train", "train_ratio": train_ratio},
        )

        test_dataset = Dataset(
            name=f"{self.name}_test",
            data=test_data,
            metadata={**self.metadata, "split": "test", "train_ratio": train_ratio},
        )

        return train_dataset, test_dataset

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format."""
        return {
            "name": self.name,
            "data": self.data,
            "metadata": self.metadata,
            "schema": self.schema,
        }

    def to_json(self, file_path: Optional[str] = None) -> str:
        """Export dataset to JSON format."""
        json_data = json.dumps(self.to_dict(), indent=2)

        if file_path:
            with open(file_path, "w") as f:
                f.write(json_data)

        return json_data

    def to_csv(self, file_path: str):
        """Export dataset to CSV format."""
        df = pd.DataFrame(self.data)
        df.to_csv(file_path, index=False)

    def validate_schema(self) -> bool:
        """Validate dataset items against schema."""
        if not self.schema:
            return True

        required_fields = self.schema.get("required", [])

        for item in self.data:
            for field in required_fields:
                if field not in item:
                    return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "size": self.size,
            "fields": list(self.data[0].keys()) if self.data else [],
            "metadata": self.metadata,
        }

        if self.data:
            for field in stats["fields"]:
                values = [item.get(field) for item in self.data if field in item]
                if values:
                    if all(isinstance(v, str) for v in values):
                        stats[f"{field}_avg_length"] = sum(
                            len(str(v)) for v in values
                        ) / len(values)
                    elif all(isinstance(v, (int, float)) for v in values):
                        stats[f"{field}_mean"] = sum(values) / len(values)
                        stats[f"{field}_min"] = min(values)
                        stats[f"{field}_max"] = max(values)

        return stats


def load_dataset(source: Union[str, Path, Dict[str, Any]], **kwargs) -> Dataset:
    """
    Load dataset from various sources.

    Args:
        source: File path, URL, or dictionary data
        **kwargs: Additional parameters for dataset creation

    Returns:
        Dataset object
    """

    if isinstance(source, dict):
        return Dataset(
            name=kwargs.get("name", "custom_dataset"),
            data=source.get("data", []),
            metadata=source.get("metadata", {}),
            schema=source.get("schema"),
        )

    elif isinstance(source, (str, Path)):
        source_path = Path(source)

        if source_path.suffix == ".json":
            with open(source_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                return Dataset(
                    name=data.get("name", source_path.stem),
                    data=data["data"],
                    metadata=data.get("metadata", {}),
                    schema=data.get("schema"),
                )
            elif isinstance(data, list):
                return Dataset(
                    name=kwargs.get("name", source_path.stem),
                    data=data,
                    metadata=kwargs.get("metadata", {}),
                )

        elif source_path.suffix == ".csv":
            df = pd.read_csv(source_path)
            data = df.to_dict("records")

            return Dataset(
                name=kwargs.get("name", source_path.stem),
                data=data,
                metadata=kwargs.get("metadata", {}),
            )

        elif str(source).startswith(("http://", "https://")):
            response = requests.get(source)
            response.raise_for_status()

            if source.endswith(".json"):
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    return Dataset(
                        name=data.get("name", "remote_dataset"),
                        data=data["data"],
                        metadata=data.get("metadata", {}),
                        schema=data.get("schema"),
                    )
                elif isinstance(data, list):
                    return Dataset(
                        name=kwargs.get("name", "remote_dataset"),
                        data=data,
                        metadata=kwargs.get("metadata", {}),
                    )

        else:
            raise ValueError(
                f"Unsupported file format '{source_path.suffix}'. Supported formats: .json, .csv"
            )


def create_qa_dataset(questions: List[str], answers: List[str], **kwargs) -> Dataset:
    """
    Create a question-answering dataset.

    Args:
        questions: List of questions
        answers: List of corresponding answers
        **kwargs: Additional metadata

    Returns:
        Dataset object
    """

    if len(questions) != len(answers):
        raise ValueError("Questions and answers must have the same length")

    data = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

    return Dataset(
        name=kwargs.get("name", "qa_dataset"),
        data=data,
        metadata={
            "task": "question_answering",
            "size": len(data),
            **kwargs.get("metadata", {}),
        },
        schema={
            "required": ["question", "answer"],
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
            },
        },
    )


def create_summarization_dataset(
    documents: List[str], summaries: List[str], **kwargs
) -> Dataset:
    """
    Create a text summarization dataset.

    Args:
        documents: List of documents to summarize
        summaries: List of corresponding summaries
        **kwargs: Additional metadata

    Returns:
        Dataset object
    """

    if len(documents) != len(summaries):
        raise ValueError("Documents and summaries must have the same length")

    data = [
        {"document": doc, "summary": summ} for doc, summ in zip(documents, summaries)
    ]

    return Dataset(
        name=kwargs.get("name", "summarization_dataset"),
        data=data,
        metadata={
            "task": "summarization",
            "size": len(data),
            **kwargs.get("metadata", {}),
        },
        schema={
            "required": ["document", "summary"],
            "properties": {
                "document": {"type": "string"},
                "summary": {"type": "string"},
            },
        },
    )


def create_classification_dataset(
    texts: List[str], labels: List[str], **kwargs
) -> Dataset:
    """
    Create a text classification dataset.

    Args:
        texts: List of texts to classify
        labels: List of corresponding labels
        **kwargs: Additional metadata

    Returns:
        Dataset object
    """

    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have the same length")

    data = [{"text": text, "label": label} for text, label in zip(texts, labels)]

    return Dataset(
        name=kwargs.get("name", "classification_dataset"),
        data=data,
        metadata={
            "task": "classification",
            "size": len(data),
            "unique_labels": list(set(labels)),
            **kwargs.get("metadata", {}),
        },
        schema={
            "required": ["text", "label"],
            "properties": {"text": {"type": "string"}, "label": {"type": "string"}},
        },
    )


class DatasetRegistry:
    """Registry for managing multiple datasets."""

    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}

    def register(self, dataset: Dataset):
        self.datasets[dataset.name] = dataset

    def get(self, name: str) -> Optional[Dataset]:
        return self.datasets.get(name)

    def list(self) -> List[str]:
        return list(self.datasets.keys())

    def remove(self, name: str):
        if name in self.datasets:
            del self.datasets[name]

    def clear(self):
        self.datasets.clear()


# Global dataset registry
registry = DatasetRegistry()


def load_mmlu_sample() -> Dataset:
    sample_data = [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "answer": "Paris",
            "subject": "geography",
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "4",
            "subject": "mathematics",
        },
    ]

    return Dataset(
        name="mmlu_sample",
        data=sample_data,
        metadata={
            "task": "multiple_choice_qa",
            "source": "MMLU",
            "description": "Sample from Massive Multitask Language Understanding",
        },
    )


def load_hellaswag_sample() -> Dataset:
    """Load a sample of HellaSwag dataset."""
    sample_data = [
        {
            "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
            "endings": [
                "rinses the bucket off with soap and blow dry the dog.",
                "uses a hose to keep the dog from getting soapy.",
                "gets the dog wet, then it runs away again.",
                "gets into the bath tub with the dog.",
            ],
            "label": 2,
        }
    ]

    return Dataset(
        name="hellaswag_sample",
        data=sample_data,
        metadata={
            "task": "sentence_completion",
            "source": "HellaSwag",
            "description": "Commonsense reasoning benchmark",
        },
    )


def load_gsm8k_sample() -> Dataset:
    """Load a sample of GSM8K (Grade School Math 8K) dataset."""
    sample_data = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much money does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs every day. She makes 9 * $2 = $18 every day at the farmers' market.",
        }
    ]

    return Dataset(
        name="gsm8k_sample",
        data=sample_data,
        metadata={
            "task": "math_word_problems",
            "source": "GSM8K",
            "description": "Grade school math word problems",
        },
    )

from typing import List, Dict, Any, Optional, Union, Callable, cast
import json
import pandas as pd
from pathlib import Path
import requests
from dataclasses import dataclass
import hashlib
import random

from .types import (
    DatasetItem,
    DatasetMetadata,
    DatasetSchema,
    DatasetDict,
    DatasetInfo,
)


def _validate_dataset_item(item: Any) -> DatasetItem:
    """
    Validate and convert a dictionary to DatasetItem.

    Args:
        item: Dictionary or any value to validate

    Returns:
        Validated DatasetItem

    Raises:
        ValueError: If item is not a dictionary
    """
    if not isinstance(item, dict):
        raise ValueError(f"Expected dict for DatasetItem, got {type(item).__name__}")
    return cast(DatasetItem, item)


def _validate_dataset_items(items: Any) -> List[DatasetItem]:
    """
    Validate and convert a list of dictionaries to List[DatasetItem].

    Args:
        items: List of dictionaries or any value to validate

    Returns:
        Validated List[DatasetItem]

    Raises:
        ValueError: If items is not a list or contains non-dict items
    """
    if not isinstance(items, list):
        raise ValueError(f"Expected list for dataset data, got {type(items).__name__}")

    validated_items: List[DatasetItem] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected dict for dataset item at index {i}, got {type(item).__name__}"
            )
        validated_items.append(cast(DatasetItem, item))

    return validated_items


def _validate_dataset_metadata(metadata: Any) -> Optional[DatasetMetadata]:
    """
    Validate and convert metadata to DatasetMetadata.

    Args:
        metadata: Dictionary or None to validate

    Returns:
        Validated DatasetMetadata or None
    """
    if metadata is None:
        return None

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Expected dict or None for DatasetMetadata, got {type(metadata).__name__}"
        )

    return cast(DatasetMetadata, metadata)


def _validate_dataset_schema(schema: Any) -> Optional[DatasetSchema]:
    """
    Validate and convert schema to DatasetSchema.

    Args:
        schema: Dictionary or None to validate

    Returns:
        Validated DatasetSchema or None
    """
    if schema is None:
        return None

    if not isinstance(schema, dict):
        raise ValueError(
            f"Expected dict or None for DatasetSchema, got {type(schema).__name__}"
        )

    return cast(DatasetSchema, schema)


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
    data: List[DatasetItem]
    metadata: Optional[DatasetMetadata] = None
    schema: Optional[DatasetSchema] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = cast(DatasetMetadata, {})

        if not self.metadata:
            self.metadata = cast(
                DatasetMetadata,
                {
                    "size": len(self.data),
                    "created_at": pd.Timestamp.now().isoformat(),
                    "hash": self._compute_hash(),
                },
            )

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

    def filter(self, condition: Callable[[DatasetItem], bool]) -> "Dataset":
        """Filter dataset items based on condition."""
        filtered_data = [item for item in self.data if condition(item)]
        metadata = self.metadata or cast(DatasetMetadata, {})
        return Dataset(
            name=f"{self.name}_filtered",
            data=filtered_data,
            metadata=cast(
                DatasetMetadata,
                {**metadata, "filtered": True, "original_size": self.size},
            ),
        )

    def sample(self, n: int, random_state: Optional[int] = None) -> "Dataset":
        """Sample n items from dataset."""
        if random_state:
            random.seed(random_state)

        sampled_data: List[DatasetItem] = random.sample(
            self.data, min(n, len(self.data))
        )
        metadata = self.metadata or cast(DatasetMetadata, {})
        return Dataset(
            name=f"{self.name}_sample_{n}",
            data=sampled_data,
            metadata=cast(
                DatasetMetadata, {**metadata, "sampled": True, "sample_size": n}
            ),
        )

    def split(
        self, train_ratio: float = 0.8, random_state: Optional[int] = None
    ) -> tuple["Dataset", "Dataset"]:
        """Split dataset into train and test sets."""
        if random_state:
            random.seed(random_state)

        shuffled_data: List[DatasetItem] = self.data.copy()
        random.shuffle(shuffled_data)

        split_idx = int(len(shuffled_data) * train_ratio)
        train_data: List[DatasetItem] = shuffled_data[:split_idx]
        test_data: List[DatasetItem] = shuffled_data[split_idx:]

        train_dataset = Dataset(
            name=f"{self.name}_train",
            data=train_data,
            metadata=cast(
                DatasetMetadata,
                {
                    **(self.metadata or cast(DatasetMetadata, {})),
                    "split": "train",
                    "train_ratio": train_ratio,
                },
            ),
        )

        test_dataset = Dataset(
            name=f"{self.name}_test",
            data=test_data,
            metadata=cast(
                DatasetMetadata,
                {
                    **(self.metadata or cast(DatasetMetadata, {})),
                    "split": "test",
                    "test_ratio": 1 - train_ratio,
                },
            ),
        )

        return train_dataset, test_dataset

    def to_dict(self) -> DatasetDict:
        """Convert dataset to dictionary format."""
        return cast(
            DatasetDict,
            {
                "name": self.name,
                "data": self.data,
                "metadata": self.metadata,
                "schema": self.schema,
            },
        )

    def to_json(self, file_path: Optional[str] = None) -> str:
        """Export dataset to JSON format."""
        json_data = json.dumps(self.to_dict(), indent=2)

        if file_path:
            with open(file_path, "w") as f:
                f.write(json_data)

        return json_data

    def to_csv(self, file_path: str) -> None:
        """Export dataset to CSV format."""
        df = pd.DataFrame(self.data)
        df.to_csv(file_path, index=False)

    def validate_schema(self) -> bool:
        """Validate dataset items against schema."""
        if not self.schema:
            return True

        # Support both "required" and "required_fields" for backward compatibility
        # Check if "required" key exists first, then fall back to "required_fields"
        if "required" in self.schema:
            required_fields = self.schema["required"]
        elif "required_fields" in self.schema:
            required_fields = self.schema["required_fields"]
        else:
            required_fields = []

        for item in self.data:
            for field in required_fields:
                if field not in item:
                    return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        fields: List[str] = list(self.data[0].keys()) if self.data else []
        stats: Dict[str, Any] = {
            "size": self.size,
            "fields": fields,
            "metadata": self.metadata,
        }

        if self.data:
            for field in fields:
                values = [item.get(field) for item in self.data if field in item]
                if values:
                    if all(isinstance(v, str) for v in values):
                        stats[f"{field}_avg_length"] = sum(
                            len(str(v)) for v in values
                        ) / len(values)
                    elif all(isinstance(v, (int, float)) for v in values):
                        # Type narrowing: we know values are numeric here
                        numeric_values = [
                            v for v in values if isinstance(v, (int, float))
                        ]
                        stats[f"{field}_mean"] = sum(numeric_values) / len(
                            numeric_values
                        )
                        stats[f"{field}_min"] = min(numeric_values)
                        stats[f"{field}_max"] = max(numeric_values)

        return stats


def load_dataset(source: Union[str, Path, DatasetDict], **kwargs: Any) -> Dataset:
    """
    Load dataset from various sources.

    Args:
        source: File path, URL, or dictionary data
        **kwargs: Additional parameters for dataset creation

    Returns:
        Dataset object
    """

    if isinstance(source, dict):
        # Type narrowing: after isinstance check, treat as DatasetDict
        # Note: .get() on TypedDict with total=False returns Any for optional keys,
        # but we know the structure from DatasetDict, so we use proper type annotations
        dataset_dict: DatasetDict = source
        # Prefer name from DatasetDict if present, otherwise fall back to kwargs
        name_from_dict: Optional[str] = dataset_dict.get("name")
        name: str = (
            name_from_dict
            if isinstance(name_from_dict, str)
            else kwargs.get("name", "custom_dataset")
        )
        data: List[DatasetItem] = dataset_dict.get("data", [])
        metadata: Optional[DatasetMetadata] = dataset_dict.get("metadata")
        schema: Optional[DatasetSchema] = dataset_dict.get("schema")

        return Dataset(
            name=name,
            data=_validate_dataset_items(data),
            metadata=_validate_dataset_metadata(metadata),
            schema=_validate_dataset_schema(schema),
        )

    elif isinstance(source, (str, Path)):
        source_path = Path(source)

        if source_path.suffix == ".json":
            with open(source_path, "r") as f:
                json_data = json.load(f)

            if isinstance(json_data, dict) and "data" in json_data:
                return Dataset(
                    name=json_data.get("name", source_path.stem)
                    if isinstance(json_data.get("name"), str)
                    else source_path.stem,
                    data=_validate_dataset_items(json_data["data"]),
                    metadata=_validate_dataset_metadata(json_data.get("metadata")),
                    schema=_validate_dataset_schema(json_data.get("schema")),
                )
            elif isinstance(json_data, list):
                return Dataset(
                    name=kwargs.get("name", source_path.stem)
                    if isinstance(kwargs.get("name"), str)
                    else source_path.stem,
                    data=_validate_dataset_items(json_data),
                    metadata=_validate_dataset_metadata(kwargs.get("metadata", {})),
                )
            else:
                raise ValueError(
                    f"Invalid JSON format in '{source_path}'. Expected a list or a dict with 'data' key."
                )

        elif source_path.suffix == ".csv":
            df = pd.read_csv(source_path)
            # Type cast: pandas to_dict returns dict[Hashable, Any] but we need dict[str, Any]
            records: List[Dict[str, Any]] = [
                cast(Dict[str, Any], dict(record)) for record in df.to_dict("records")
            ]
            csv_data: List[DatasetItem] = [
                cast(DatasetItem, record) for record in records
            ]

            return Dataset(
                name=kwargs.get("name", source_path.stem)
                if isinstance(kwargs.get("name"), str)
                else source_path.stem,
                data=csv_data,
                metadata=_validate_dataset_metadata(kwargs.get("metadata")),
            )

        elif str(source).startswith(("http://", "https://")):
            # Convert to str for requests.get
            source_str = str(source)
            response = requests.get(source_str)
            response.raise_for_status()

            if source_str.endswith(".json"):
                json_data = response.json()
                if isinstance(json_data, dict) and "data" in json_data:
                    return Dataset(
                        name=json_data.get("name", "remote_dataset")
                        if isinstance(json_data.get("name"), str)
                        else "remote_dataset",
                        data=_validate_dataset_items(json_data["data"]),
                        metadata=_validate_dataset_metadata(json_data.get("metadata")),
                        schema=_validate_dataset_schema(json_data.get("schema")),
                    )
                elif isinstance(json_data, list):
                    return Dataset(
                        name=kwargs.get("name", "remote_dataset")
                        if isinstance(kwargs.get("name"), str)
                        else "remote_dataset",
                        data=_validate_dataset_items(json_data),
                        metadata=_validate_dataset_metadata(kwargs.get("metadata", {})),
                    )
                else:
                    raise ValueError(
                        f"Invalid JSON format from '{source_str}'. Expected a list or a dict with 'data' key."
                    )
            else:
                raise ValueError(
                    f"Unsupported URL format '{source_str}'. Only .json URLs are supported."
                )

        else:
            raise ValueError(
                f"Unsupported file format '{source_path.suffix}'. Supported formats: .json, .csv"
            )

    raise ValueError(f"Unable to load dataset from source: {source}")


def create_qa_dataset(
    questions: List[str], answers: List[str], **kwargs: Any
) -> Dataset:
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

    data: List[DatasetItem] = [
        cast(DatasetItem, {"question": q, "answer": a})
        for q, a in zip(questions, answers)
    ]

    return Dataset(
        name=kwargs.get("name", "qa_dataset")
        if isinstance(kwargs.get("name"), str)
        else "qa_dataset",
        data=data,
        metadata=cast(
            DatasetMetadata,
            {
                "task": "question_answering",
                "size": len(data),
                **kwargs.get("metadata", {}),
            },
        ),
        schema=cast(
            DatasetSchema,
            {
                "required": ["question", "answer"],
                "prompt_field": "question",
                "reference_field": "answer",
            },
        ),
    )


def create_summarization_dataset(
    documents: List[str], summaries: List[str], **kwargs: Any
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

    data: List[DatasetItem] = [
        cast(DatasetItem, {"document": doc, "summary": summ})
        for doc, summ in zip(documents, summaries)
    ]

    return Dataset(
        name=kwargs.get("name", "summarization_dataset")
        if isinstance(kwargs.get("name"), str)
        else "summarization_dataset",
        data=data,
        metadata=cast(
            DatasetMetadata,
            {
                "task": "summarization",
                "size": len(data),
                **kwargs.get("metadata", {}),
            },
        ),
        schema=cast(
            DatasetSchema,
            {
                "required": ["document", "summary"],
                "prompt_field": "document",
                "reference_field": "summary",
            },
        ),
    )


def create_classification_dataset(
    texts: List[str], labels: List[str], **kwargs: Any
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

    data: List[DatasetItem] = [
        cast(DatasetItem, {"text": text, "label": label})
        for text, label in zip(texts, labels)
    ]

    return Dataset(
        name=kwargs.get("name", "classification_dataset")
        if isinstance(kwargs.get("name"), str)
        else "classification_dataset",
        data=data,
        metadata=cast(
            DatasetMetadata,
            {
                "task": "classification",
                "size": len(data),
                "unique_labels": list(set(labels)),
                **kwargs.get("metadata", {}),
            },
        ),
        schema=cast(
            DatasetSchema,
            {
                "required": ["text", "label"],
                "prompt_field": "text",
                "reference_field": "label",
            },
        ),
    )


class DatasetRegistry:
    """Registry for managing multiple datasets."""

    def __init__(self) -> None:
        self.datasets: Dict[str, Dataset] = {}

    def register(self, dataset: Dataset) -> None:
        self.datasets[dataset.name] = dataset

    def get(self, name: str) -> Optional[Dataset]:
        return self.datasets.get(name)

    def list(self) -> List[str]:
        return list(self.datasets.keys())

    def remove(self, name: str) -> None:
        if name in self.datasets:
            del self.datasets[name]

    def clear(self) -> None:
        self.datasets.clear()


# Global dataset registry
registry = DatasetRegistry()


def load_mmlu_sample() -> Dataset:
    sample_data: List[DatasetItem] = [
        cast(
            DatasetItem,
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "Paris",
                "subject": "geography",
            },
        ),
        cast(
            DatasetItem,
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "4",
                "subject": "mathematics",
            },
        ),
    ]

    return Dataset(
        name="mmlu_sample",
        data=sample_data,
        metadata=cast(
            DatasetMetadata,
            {
                "task": "multiple_choice_qa",
                "source": "MMLU",
                "description": "Sample from Massive Multitask Language Understanding",
            },
        ),
    )


def load_hellaswag_sample() -> Dataset:
    """Load a sample of HellaSwag dataset."""
    sample_data: List[DatasetItem] = [
        cast(
            DatasetItem,
            {
                "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
                "endings": [
                    "rinses the bucket off with soap and blow dry the dog.",
                    "uses a hose to keep the dog from getting soapy.",
                    "gets the dog wet, then it runs away again.",
                    "gets into the bath tub with the dog.",
                ],
                "label": 2,
            },
        )
    ]

    return Dataset(
        name="hellaswag_sample",
        data=sample_data,
        metadata=cast(
            DatasetMetadata,
            {
                "task": "sentence_completion",
                "source": "HellaSwag",
                "description": "Commonsense reasoning benchmark",
            },
        ),
    )


def load_gsm8k_sample() -> Dataset:
    """Load a sample of GSM8K (Grade School Math 8K) dataset."""
    sample_data: List[DatasetItem] = [
        cast(
            DatasetItem,
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much money does she make every day at the farmers' market?",
                "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs every day. She makes 9 * $2 = $18 every day at the farmers' market.",
            },
        )
    ]

    return Dataset(
        name="gsm8k_sample",
        data=sample_data,
        metadata=cast(
            DatasetMetadata,
            {
                "task": "math_word_problems",
                "source": "GSM8K",
                "description": "Grade school math word problems",
            },
        ),
    )


def convert_metadata_to_info(metadata: DatasetMetadata) -> DatasetInfo:
    """
    Convert DatasetMetadata to DatasetInfo for evaluation results.

    This function properly converts dataset metadata (which is stored with the dataset)
    to dataset info (which is used in evaluation results). It handles missing fields
    and ensures type safety.

    Args:
        metadata: Dataset metadata to convert

    Returns:
        DatasetInfo with converted fields
    """
    # Extract fields that exist in DatasetMetadata
    size: int = metadata.get("size", 0)
    tags: List[str] = metadata.get("tags", [])
    source: Optional[str] = metadata.get("source")
    name: Optional[str] = metadata.get("name")
    description: Optional[str] = metadata.get("description")
    version: Optional[str] = metadata.get("version")
    created_at: Optional[str] = metadata.get("created_at")

    # Extract fields that might exist but aren't in DatasetMetadata TypedDict
    # These could be present at runtime even if not in the type definition
    metadata_dict: Dict[str, Any] = cast(Dict[str, Any], metadata)
    hash_value: Optional[str] = metadata_dict.get("hash")
    task: Optional[str] = metadata_dict.get("task")
    difficulty: Optional[str] = metadata_dict.get("difficulty")

    # Build DatasetInfo with proper types
    info: DatasetInfo = {
        "size": size,
        "tags": tags,
        "source": source,
        "name": name,
        "description": description,
        "version": version,
        "created_at": created_at,
        "hash": hash_value,
        "task": task if task else "general",
    }

    # Add difficulty if available
    if difficulty:
        info["difficulty"] = difficulty

    return info

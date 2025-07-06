"""
Tests for dataset functionality
"""

import pytest
import json
import pandas as pd
from pathlib import Path
import tempfile

from benchwise.datasets import (
    Dataset,
    load_dataset,
    create_qa_dataset,
    create_summarization_dataset,
    create_classification_dataset,
    DatasetRegistry,
    registry,
    load_mmlu_sample,
    load_hellaswag_sample,
    load_gsm8k_sample,
)


class TestDataset:
    def test_dataset_creation(self, sample_qa_data):
        data = [
            {"question": q, "answer": a}
            for q, a in zip(sample_qa_data["questions"], sample_qa_data["answers"])
        ]

        dataset = Dataset(name="test_dataset", data=data, metadata={"task": "qa"})

        assert dataset.name == "test_dataset"
        assert len(dataset.data) == 5
        assert dataset.metadata["task"] == "qa"
        assert dataset.size == 5

    def test_dataset_prompts_property(self, sample_dataset):
        prompts = sample_dataset.prompts
        assert len(prompts) == 5
        assert "What is the capital of France?" in prompts
        assert "What is 2 + 2?" in prompts

    def test_dataset_references_property(self, sample_dataset):
        references = sample_dataset.references
        assert len(references) == 5
        assert "Paris" in references
        assert "4" in references

    def test_dataset_filter(self, sample_dataset):
        filtered = sample_dataset.filter(lambda x: len(x["question"]) > 15)

        assert filtered.size < sample_dataset.size
        assert filtered.name == "test_qa_dataset_filtered"
        assert filtered.metadata["filtered"]
        assert filtered.metadata["original_size"] == sample_dataset.size

    def test_dataset_sample(self, sample_dataset):
        sampled = sample_dataset.sample(n=3, random_state=42)

        assert sampled.size == 3
        assert sampled.name == "test_qa_dataset_sample_3"
        assert sampled.metadata["sampled"]
        assert sampled.metadata["sample_size"] == 3

        # Test sampling more than available
        large_sample = sample_dataset.sample(n=10, random_state=42)
        assert large_sample.size == sample_dataset.size  # Should return all

    def test_dataset_split(self, sample_dataset):
        train, test = sample_dataset.split(train_ratio=0.6, random_state=42)

        assert train.name == "test_qa_dataset_train"
        assert test.name == "test_qa_dataset_test"
        assert train.size + test.size == sample_dataset.size
        assert train.metadata["split"] == "train"
        assert test.metadata["split"] == "test"
        assert train.metadata["train_ratio"] == 0.6

    def test_dataset_to_dict(self, sample_dataset):
        dataset_dict = sample_dataset.to_dict()

        assert "name" in dataset_dict
        assert "data" in dataset_dict
        assert "metadata" in dataset_dict
        assert dataset_dict["name"] == sample_dataset.name
        assert len(dataset_dict["data"]) == sample_dataset.size

    def test_dataset_to_json(self, sample_dataset):
        json_str = sample_dataset.to_json()

        parsed = json.loads(json_str)
        assert parsed["name"] == sample_dataset.name
        assert len(parsed["data"]) == sample_dataset.size

    def test_dataset_to_csv(self, sample_dataset):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            sample_dataset.to_csv(temp_path)

            # Verify CSV was created and is readable
            df = pd.read_csv(temp_path)
            assert len(df) == sample_dataset.size
            assert "question" in df.columns
            assert "answer" in df.columns

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_dataset_validate_schema(self):
        valid_data = [{"question": "Q1", "answer": "A1"}]
        valid_dataset = Dataset(
            name="valid", data=valid_data, schema={"required": ["question", "answer"]}
        )
        assert valid_dataset.validate_schema()

        # Invalid dataset
        invalid_data = [{"question": "Q1"}]  # Missing answer
        invalid_dataset = Dataset(
            name="invalid",
            data=invalid_data,
            schema={"required": ["question", "answer"]},
        )
        assert not invalid_dataset.validate_schema()

    def test_dataset_statistics(self, sample_dataset):
        stats = sample_dataset.get_statistics()

        assert stats["size"] == 5
        assert "question" in stats["fields"]
        assert "answer" in stats["fields"]
        assert "question_avg_length" in stats
        assert "answer_avg_length" in stats


class TestDatasetLoading:
    def test_load_dataset_from_dict(self, sample_qa_data):
        data_dict = {
            "data": [
                {"question": q, "answer": a}
                for q, a in zip(sample_qa_data["questions"], sample_qa_data["answers"])
            ],
            "metadata": {"task": "qa"},
        }

        dataset = load_dataset(data_dict, name="dict_dataset")

        assert dataset.name == "dict_dataset"
        assert len(dataset.data) == 5
        assert dataset.metadata["task"] == "qa"

    def test_load_dataset_from_json_file(self, temp_dataset_file):
        dataset = load_dataset(temp_dataset_file)

        assert dataset.size == 5
        assert len(dataset.prompts) == 5
        assert len(dataset.references) == 5

    def test_load_dataset_from_csv_file(self, temp_csv_dataset_file):
        dataset = load_dataset(temp_csv_dataset_file)

        assert dataset.size == 5
        assert len(dataset.prompts) == 5
        assert len(dataset.references) == 5

    def test_load_dataset_invalid_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Invalid format")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_dataset(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDatasetCreators:
    def test_create_qa_dataset(self, sample_qa_data):
        dataset = create_qa_dataset(
            questions=sample_qa_data["questions"],
            answers=sample_qa_data["answers"],
            name="qa_test",
        )

        assert dataset.name == "qa_test"
        assert dataset.metadata["task"] == "question_answering"
        assert len(dataset.data) == 5
        assert dataset.schema["required"] == ["question", "answer"]

    def test_create_qa_dataset_mismatched_lengths(self, sample_qa_data):
        with pytest.raises(
            ValueError, match="Questions and answers must have the same length"
        ):
            create_qa_dataset(
                questions=sample_qa_data["questions"],
                answers=sample_qa_data["answers"][:3],  # Shorter list
            )

    def test_create_summarization_dataset(self):
        documents = ["Long document 1", "Long document 2"]
        summaries = ["Summary 1", "Summary 2"]

        dataset = create_summarization_dataset(
            documents=documents, summaries=summaries, name="sum_test"
        )

        assert dataset.name == "sum_test"
        assert dataset.metadata["task"] == "summarization"
        assert len(dataset.data) == 2
        assert dataset.schema["required"] == ["document", "summary"]

    def test_create_classification_dataset(self):
        texts = ["Positive text", "Negative text", "Neutral text"]
        labels = ["positive", "negative", "neutral"]

        dataset = create_classification_dataset(
            texts=texts, labels=labels, name="cls_test"
        )

        assert dataset.name == "cls_test"
        assert dataset.metadata["task"] == "classification"
        assert set(dataset.metadata["unique_labels"]) == set(
            ["positive", "negative", "neutral"]
        )
        assert len(dataset.data) == 3


class TestDatasetRegistry:
    def test_registry_register_and_get(self, sample_dataset):
        registry_test = DatasetRegistry()

        registry_test.register(sample_dataset)
        retrieved = registry_test.get(sample_dataset.name)

        assert retrieved == sample_dataset
        assert sample_dataset.name in registry_test.list()

    def test_registry_remove(self, sample_dataset):
        registry_test = DatasetRegistry()

        registry_test.register(sample_dataset)
        assert sample_dataset.name in registry_test.list()

        registry_test.remove(sample_dataset.name)
        assert sample_dataset.name not in registry_test.list()
        assert registry_test.get(sample_dataset.name) is None

    def test_registry_clear(self, sample_dataset):
        registry_test = DatasetRegistry()

        registry_test.register(sample_dataset)
        assert len(registry_test.list()) == 1

        registry_test.clear()
        assert len(registry_test.list()) == 0

    def test_global_registry(self, sample_dataset):
        initial_count = len(registry.list())

        registry.register(sample_dataset)
        assert len(registry.list()) == initial_count + 1

        # Cleanup
        registry.remove(sample_dataset.name)


class TestSampleDatasets:
    """Test built-in sample datasets"""

    def test_load_mmlu_sample(self):
        dataset = load_mmlu_sample()

        assert dataset.name == "mmlu_sample"
        assert dataset.metadata["task"] == "multiple_choice_qa"
        assert dataset.metadata["source"] == "MMLU"
        assert dataset.size >= 1

        first_item = dataset.data[0]
        assert "question" in first_item
        assert "choices" in first_item
        assert "answer" in first_item
        assert "subject" in first_item

    def test_load_hellaswag_sample(self):
        dataset = load_hellaswag_sample()

        assert dataset.name == "hellaswag_sample"
        assert dataset.metadata["task"] == "sentence_completion"
        assert dataset.metadata["source"] == "HellaSwag"
        assert dataset.size >= 1

        first_item = dataset.data[0]
        assert "context" in first_item
        assert "endings" in first_item
        assert "label" in first_item

    def test_load_gsm8k_sample(self):
        dataset = load_gsm8k_sample()

        assert dataset.name == "gsm8k_sample"
        assert dataset.metadata["task"] == "math_word_problems"
        assert dataset.metadata["source"] == "GSM8K"
        assert dataset.size >= 1

        first_item = dataset.data[0]
        assert "question" in first_item
        assert "answer" in first_item


class TestDatasetEdgeCases:
    def test_empty_dataset(self):
        dataset = Dataset(name="empty", data=[])

        assert dataset.size == 0
        assert dataset.prompts == []
        assert dataset.references == []
        assert dataset.get_statistics()["size"] == 0

    def test_dataset_with_missing_fields(self):
        data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2"},
            {"answer": "A3"},
        ]

        dataset = Dataset(name="mixed", data=data)

        prompts = dataset.prompts
        references = dataset.references

        assert len(prompts) == 2
        assert len(references) == 2

    def test_dataset_hash_consistency(self, sample_dataset):
        hash1 = sample_dataset._compute_hash()
        hash2 = sample_dataset._compute_hash()
        assert hash1 == hash2

        # Hash should change with different data
        different_dataset = Dataset(name="different", data=[{"a": 1}])
        hash3 = different_dataset._compute_hash()
        assert hash1 != hash3

    def test_dataset_with_unicode(self):
        data = [
            {"question": "What is café in English?", "answer": "coffee"},
            {"question": "Translate 漢字", "answer": "Chinese characters"},
        ]

        dataset = Dataset(name="unicode", data=data)

        assert dataset.size == 2
        assert len(dataset.prompts) == 2
        assert "café" in dataset.prompts[0]
        assert "漢字" in dataset.prompts[1]

"""
Test fixtures and utilities
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from benchwise.datasets import create_qa_dataset
from benchwise.results import EvaluationResult, BenchmarkResult
from benchwise.config import BenchWiseConfig


@pytest.fixture
def sample_qa_data():
    return {
        "questions": [
            "What is the capital of France?",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet?",
            "What year did WWII end?",
        ],
        "answers": ["Paris", "4", "William Shakespeare", "Jupiter", "1945"],
    }


@pytest.fixture
def sample_dataset(sample_qa_data):
    return create_qa_dataset(
        questions=sample_qa_data["questions"],
        answers=sample_qa_data["answers"],
        name="test_qa_dataset",
    )


@pytest.fixture
def sample_responses():
    return ["Paris", "4", "Shakespeare", "Jupiter", "1945"]


@pytest.fixture
def sample_references():
    return ["Paris", "4", "William Shakespeare", "Jupiter", "1945"]


@pytest.fixture
def sample_evaluation_result():
    return EvaluationResult(
        model_name="test-model",
        test_name="test_evaluation",
        result={"accuracy": 0.8, "f1": 0.75},
        duration=1.5,
        dataset_info={"size": 5, "task": "qa"},
        metadata={"temperature": 0.0},
    )


@pytest.fixture
def sample_benchmark_result(sample_evaluation_result):
    result = BenchmarkResult(
        benchmark_name="test_benchmark", metadata={"description": "Test benchmark"}
    )
    result.add_result(sample_evaluation_result)

    # Add a failed result
    failed_result = EvaluationResult(
        model_name="failed-model",
        test_name="test_evaluation",
        error="Test error",
        duration=0.0,
        dataset_info={"size": 5, "task": "qa"},
    )
    result.add_result(failed_result)

    return result


@pytest.fixture
def temp_dataset_file(sample_qa_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = [
            {"question": q, "answer": a}
            for q, a in zip(sample_qa_data["questions"], sample_qa_data["answers"])
        ]
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_csv_dataset_file(sample_qa_data):
    import pandas as pd

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "question": sample_qa_data["questions"],
                "answer": sample_qa_data["answers"],
            }
        )
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def test_config():
    return BenchWiseConfig(
        api_url="http://localhost:8000",
        upload_enabled=False,
        cache_enabled=False,
        debug=True,
        offline_mode=True,
    )


@pytest.fixture
def mock_model_responses():
    return {
        "gpt-3.5-turbo": [
            "Paris is the capital of France.",
            "The answer is 4.",
            "William Shakespeare wrote Romeo and Juliet.",
            "Jupiter is the largest planet in our solar system.",
            "World War II ended in 1945.",
        ],
        "claude-3-haiku": ["Paris", "4", "Shakespeare", "Jupiter", "1945"],
        "mock-test": [
            "Mock response for: What is the capital of France?",
            "Mock response for: What is 2 + 2?",
            "Mock response for: Who wrote Romeo and Juliet?",
            "Mock response for: What is the largest planet?",
            "Mock response for: What year did WWII end?",
        ],
    }


@pytest.fixture
def metrics_test_data():
    return {
        "perfect_match": {
            "predictions": ["Hello world", "Python is great", "AI is amazing"],
            "references": ["Hello world", "Python is great", "AI is amazing"],
            "expected_accuracy": 1.0,
        },
        "partial_match": {
            "predictions": ["Hello world", "Python rocks", "AI is cool"],
            "references": ["Hello world", "Python is great", "AI is amazing"],
            "expected_accuracy": 1 / 3,  # Only first one matches exactly
        },
        "no_match": {
            "predictions": ["Goodbye", "Java", "ML"],
            "references": ["Hello world", "Python is great", "AI is amazing"],
            "expected_accuracy": 0.0,
        },
    }


# Note: event_loop fixture removed - using pytest-asyncio auto mode


class MockModelAdapter:
    def __init__(self, model_name: str, responses: List[str] = None):
        self.model_name = model_name
        self.responses = responses or ["Mock response"] * 10
        self.call_count = 0

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        self.call_count += 1
        if len(prompts) <= len(self.responses):
            return self.responses[: len(prompts)]
        else:
            repeated = self.responses * ((len(prompts) // len(self.responses)) + 1)
            return repeated[: len(prompts)]

    def get_token_count(self, text: str) -> int:
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        return 0.001


def create_temp_cache_dir():
    import tempfile

    return tempfile.mkdtemp(prefix="benchwise_test_cache_")


def cleanup_temp_dir(temp_dir: str):
    import shutil

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@pytest.fixture
def temp_cache_dir():
    temp_dir = create_temp_cache_dir()
    yield temp_dir
    cleanup_temp_dir(temp_dir)


def assert_metric_result_structure(result: Dict[str, Any], metric_name: str):
    assert isinstance(result, dict), f"{metric_name} should return a dictionary"
    assert "scores" in result, f"{metric_name} should include individual scores"
    assert isinstance(result["scores"], list), f"{metric_name} scores should be a list"


def assert_evaluation_result_valid(result: EvaluationResult):
    assert isinstance(result, EvaluationResult)
    assert result.model_name is not None
    assert result.test_name is not None
    assert isinstance(result.duration, (int, float))
    assert result.duration >= 0
    assert isinstance(result.timestamp, datetime)


def assert_benchmark_result_valid(result: BenchmarkResult):
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_name is not None
    assert isinstance(result.results, list)
    assert isinstance(result.metadata, dict)
    assert isinstance(result.timestamp, datetime)

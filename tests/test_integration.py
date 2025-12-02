"""
Integration tests for Benchwise SDK

These tests verify that the main components work together correctly.
"""

import pytest
from unittest.mock import patch, AsyncMock

from benchwise import (
    evaluate,
    benchmark,
    create_qa_dataset,
    accuracy,
    get_model_adapter,
)
from benchwise.config import get_api_config
from tests.conftest import MockModelAdapter


class TestBasicIntegration:
    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self):
        dataset = create_qa_dataset(
            questions=["What is 2+2?", "What is the capital of France?"],
            answers=["4", "Paris"],
            name="integration_test",
        )

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("mock-test", ["4", "Paris"])
            mock_get_adapter.return_value = mock_adapter

            @evaluate("mock-test")
            async def test_evaluation(model, dataset):
                responses = await model.generate(dataset.prompts)
                acc_result = accuracy(responses, dataset.references)
                return {"accuracy": acc_result["accuracy"]}

            results = await test_evaluation(dataset)

            assert len(results) == 1
            result = results[0]
            assert result.success
            assert result.model_name == "mock-test"
            assert result.result["accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_benchmark_with_multiple_models(self):
        dataset = create_qa_dataset(
            questions=["Test question"],
            answers=["Test answer"],
            name="multi_model_test",
        )

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:

            def get_adapter(model_name):
                return MockModelAdapter(model_name, ["Test answer"])

            mock_get_adapter.side_effect = get_adapter

            @benchmark("multi_model_benchmark", "Test with multiple models")
            @evaluate("mock-model-1", "mock-model-2")
            async def multi_model_test(model, dataset):
                responses = await model.generate(dataset.prompts)
                acc_result = accuracy(responses, dataset.references)
                return {"accuracy": acc_result["accuracy"]}

            results = await multi_model_test(dataset)

            assert len(results) == 2
            assert all(r.success for r in results)
            # Check that benchmark metadata exists
            assert "name" in results[0].metadata
            assert results[0].metadata["name"] == "multi_model_benchmark"
            assert results[0].metadata["description"] == "Test with multiple models"

    def test_dataset_and_metrics_integration(self):
        dataset = create_qa_dataset(
            questions=["Question 1", "Question 2"],
            answers=["Answer 1", "Answer 2"],
            name="metrics_test",
        )

        # Test dataset properties
        assert len(dataset.prompts) == 2
        assert len(dataset.references) == 2

        # Test metrics with dataset data
        predictions = ["Answer 1", "Different answer"]
        references = dataset.references

        acc_result = accuracy(predictions, references)
        assert acc_result["accuracy"] == 0.5
        assert acc_result["correct"] == 1
        assert acc_result["total"] == 2

    def test_model_factory_integration(self):
        model_types = [
            ("gpt-3.5-turbo", "OpenAIAdapter"),
            ("claude-3-haiku", "AnthropicAdapter"),
            ("gemini-pro", "GoogleAdapter"),
            ("mock-test", "MockAdapter"),
        ]

        for model_name, expected_type in model_types:
            adapter = get_model_adapter(model_name)
            assert adapter.__class__.__name__ == expected_type
            assert adapter.model_name == model_name

        with (
            patch("transformers.AutoTokenizer"),
            patch("transformers.AutoModelForCausalLM"),
        ):
            adapter = get_model_adapter("test/unknown-model")
            assert adapter.__class__.__name__ == "HuggingFaceAdapter"
            assert adapter.model_name == "test/unknown-model"


class TestErrorHandlingIntegration:
    @pytest.mark.asyncio
    async def test_evaluation_with_model_error(self):
        dataset = create_qa_dataset(
            questions=["Test"], answers=["Answer"], name="error_test"
        )

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            # Mock adapter that fails
            failing_adapter = MockModelAdapter("failing-model")
            failing_adapter.generate = AsyncMock(side_effect=Exception("Model error"))
            mock_get_adapter.return_value = failing_adapter

            @evaluate("failing-model")
            async def failing_evaluation(model, dataset):
                await model.generate(dataset.prompts)
                return {"score": 1.0}

            # Run evaluation
            results = await failing_evaluation(dataset)

            # Should have error result
            assert len(results) == 1
            assert results[0].failed
            assert "Model error" in results[0].error

    def test_metrics_with_empty_data(self):
        acc_result = accuracy([], [])
        assert acc_result["accuracy"] == 0.0
        assert acc_result["total"] == 0

        # Test metrics with mismatched lengths
        with pytest.raises(ValueError):
            accuracy(["a"], ["a", "b"])

    def test_dataset_edge_cases(self):
        empty_dataset = create_qa_dataset([], [], name="empty")
        assert empty_dataset.size == 0
        assert empty_dataset.prompts == []
        assert empty_dataset.references == []
        single_dataset = create_qa_dataset(["Q"], ["A"], name="single")
        assert single_dataset.size == 1
        assert len(single_dataset.prompts) == 1


class TestDataFlowIntegration:
    def test_dataset_to_results_flow(self, sample_dataset):
        from benchwise.results import EvaluationResult, BenchmarkResult

        result = EvaluationResult(
            model_name="test-model",
            test_name="flow_test",
            result={"accuracy": 0.8},
            dataset_info=sample_dataset.metadata,
        )

        benchmark = BenchmarkResult("flow_benchmark")
        benchmark.add_result(result)

        assert benchmark.benchmark_name == "flow_benchmark"
        assert len(benchmark.results) == 1
        assert benchmark.results[0].dataset_info == sample_dataset.metadata

    def test_config_to_evaluation_flow(self):
        from benchwise.config import BenchwiseConfig, set_api_config

        test_config = BenchwiseConfig(upload_enabled=False, debug=True)

        original_config = get_api_config()
        set_api_config(test_config)

        try:
            current_config = get_api_config()
            assert not current_config.upload_enabled
            assert current_config.debug

        finally:
            set_api_config(original_config)


class TestEndToEndBasic:
    @pytest.mark.asyncio
    async def test_simple_qa_evaluation_e2e(self):
        dataset = create_qa_dataset(
            questions=["What is Python?", "What is 1+1?"],
            answers=["Programming language", "2"],
            name="simple_qa",
        )

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("test-model", ["Programming language", "2"])
            mock_get_adapter.return_value = mock_adapter

            @benchmark("Simple QA Test", "Basic QA evaluation")
            @evaluate("test-model")
            async def simple_qa_eval(model, dataset):
                responses = await model.generate(dataset.prompts)
                acc = accuracy(responses, dataset.references)
                return {
                    "accuracy": acc["accuracy"],
                    "total_questions": len(dataset.prompts),
                }

            results = await simple_qa_eval(dataset)

            assert len(results) == 1
            result = results[0]
            assert result.success
            assert result.result["accuracy"] == 1.0
            assert result.result["total_questions"] == 2
            assert "name" in result.metadata
            assert result.metadata["name"] == "Simple QA Test"

    def test_import_all_main_components(self):
        from benchwise import (
            evaluate,
            benchmark,
            create_qa_dataset,
            accuracy,
            get_model_adapter,
            get_api_config,
            configure_benchwise,
        )

        assert callable(evaluate)
        assert callable(benchmark)
        assert callable(create_qa_dataset)
        assert callable(accuracy)
        assert callable(get_model_adapter)
        assert callable(configure_benchwise)

        config = get_api_config()
        assert config is not None

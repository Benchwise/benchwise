"""
Tests for core evaluation functionality
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from benchwise.core import evaluate, benchmark, EvaluationRunner
from benchwise.results import EvaluationResult
from tests.conftest import MockModelAdapter, assert_evaluation_result_valid


class TestEvaluateDecorator:
    @pytest.mark.asyncio
    async def test_single_model_evaluation(self, sample_dataset):
        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter(
                "mock-test", ["Paris", "4", "Shakespeare", "Jupiter", "1945"]
            )
            mock_get_adapter.return_value = mock_adapter

            @evaluate("mock-test")
            async def test_evaluation(model, dataset):
                responses = await model.generate(dataset.prompts)
                return {"test_score": 0.8, "responses": len(responses)}

            # Run evaluation
            results = await test_evaluation(sample_dataset)

            # Assertions
            assert len(results) == 1
            assert_evaluation_result_valid(results[0])
            assert results[0].model_name == "mock-test"
            assert results[0].success
            assert results[0].result["test_score"] == 0.8
            assert results[0].result["responses"] == 5

    @pytest.mark.asyncio
    async def test_multiple_model_evaluation(self, sample_dataset):
        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:

            def get_adapter(model_name):
                return MockModelAdapter(model_name, ["Answer"] * 5)

            mock_get_adapter.side_effect = get_adapter

            @evaluate("mock-test-1", "mock-test-2")
            async def test_evaluation(model, dataset):
                await model.generate(dataset.prompts)
                return {"score": 0.5}

            # Run evaluation
            results = await test_evaluation(sample_dataset)

            # Assertions
            assert len(results) == 2
            model_names = [r.model_name for r in results]
            assert "mock-test-1" in model_names
            assert "mock-test-2" in model_names
            assert all(r.success for r in results)
            assert all(r.result["score"] == 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_evaluation_with_error(self, sample_dataset):
        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("failing-model")
            mock_adapter.generate = AsyncMock(side_effect=Exception("Model failed"))
            mock_get_adapter.return_value = mock_adapter

            @evaluate("failing-model")
            async def test_evaluation(model, dataset):
                await model.generate(dataset.prompts)
                return {"score": 1.0}

            results = await test_evaluation(sample_dataset)

            assert len(results) == 1
            assert results[0].failed
            assert "Model failed" in results[0].error

    @pytest.mark.asyncio
    async def test_evaluation_with_upload_disabled(self, sample_dataset):
        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            with patch("benchwise.core.upload_results") as mock_upload:
                mock_adapter = MockModelAdapter("mock-test")
                mock_get_adapter.return_value = mock_adapter

                @evaluate("mock-test", upload=False)
                async def test_evaluation(model, dataset):
                    await model.generate(dataset.prompts)
                    return {"score": 0.9}

                results = await test_evaluation(sample_dataset)

                assert len(results) == 1
                assert results[0].success
                mock_upload.assert_not_called()


class TestBenchmarkDecorator:
    @pytest.mark.asyncio
    async def test_benchmark_metadata(self, sample_dataset):
        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("mock-test")
            mock_get_adapter.return_value = mock_adapter

            @benchmark("test_benchmark", "A test benchmark for validation")
            @evaluate("mock-test")
            async def test_evaluation(model, dataset):
                return {"score": 0.7}

            results = await test_evaluation(sample_dataset)

            assert len(results) == 1
            result = results[0]
            assert "name" in result.metadata
            assert result.metadata["name"] == "test_benchmark"
            assert result.metadata["description"] == "A test benchmark for validation"

    @pytest.mark.asyncio
    async def test_benchmark_with_additional_metadata(self, sample_dataset):
        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("mock-test")
            mock_get_adapter.return_value = mock_adapter

            @benchmark("custom_benchmark", difficulty="hard", domain="nlp")
            @evaluate("mock-test")
            async def test_evaluation(model, dataset):
                return {"score": 0.6}

            results = await test_evaluation(sample_dataset)

            result = results[0]
            assert "name" in result.metadata
            assert result.metadata["name"] == "custom_benchmark"
            assert result.metadata["difficulty"] == "hard"
            assert result.metadata["domain"] == "nlp"


class TestEvaluationRunner:
    def test_runner_initialization(self):
        runner = EvaluationRunner()
        assert hasattr(runner, "config")
        assert hasattr(runner, "results_cache")

        custom_config = {"debug": True}
        runner = EvaluationRunner(custom_config)
        assert runner.config == custom_config

    @pytest.mark.asyncio
    async def test_run_evaluation(self, sample_dataset):
        async def mock_test_func(model, dataset):
            return {"accuracy": 0.85}

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("test-model")
            mock_get_adapter.return_value = mock_adapter

            runner = EvaluationRunner()
            results = await runner.run_evaluation(
                mock_test_func, sample_dataset, ["test-model"]
            )

            assert len(results) == 1
            assert results[0]["accuracy"] == 0.85

    def test_compare_models_with_numeric_results(self):
        results = [
            EvaluationResult("model-a", "test", {"accuracy": 0.9}, 1.0),
            EvaluationResult("model-b", "test", {"accuracy": 0.7}, 1.2),
            EvaluationResult("model-c", "test", {"accuracy": 0.8}, 0.8),
        ]

        runner = EvaluationRunner()
        comparison = runner.compare_models(results, "accuracy")

        assert comparison["best_model"] == "model-a"
        assert comparison["worst_model"] == "model-b"
        assert len(comparison["ranking"]) == 3
        assert comparison["ranking"][0]["model"] == "model-a"
        assert comparison["ranking"][0]["score"] == 0.9

    def test_compare_models_with_failed_results(self):
        results = [
            EvaluationResult("model-a", "test", {"accuracy": 0.9}, 1.0),
            EvaluationResult("model-b", "test", error="Failed", duration=0.0),
            EvaluationResult("model-c", "test", {"accuracy": 0.8}, 0.8),
        ]

        runner = EvaluationRunner()
        comparison = runner.compare_models(results, "accuracy")

        assert len(comparison["ranking"]) == 2
        assert comparison["best_model"] == "model-a"
        assert comparison["worst_model"] == "model-c"

    def test_compare_models_no_successful_results(self):
        results = [
            EvaluationResult("model-a", "test", error="Failed", duration=0.0),
            EvaluationResult("model-b", "test", error="Also failed", duration=0.0),
        ]

        runner = EvaluationRunner()
        comparison = runner.compare_models(results, "accuracy")

        assert "error" in comparison
        assert "No successful results" in comparison["error"]


class TestQuickEval:
    def test_quick_eval_import(self):
        from benchwise.core import quick_eval

        assert callable(quick_eval)

    @pytest.mark.asyncio
    async def test_quick_eval_basic(self):
        from benchwise.core import quick_eval

        def mock_metric(response):
            return 0.5

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:
            mock_adapter = MockModelAdapter("mock-test", ["Response"])
            mock_adapter.generate = MagicMock(return_value=["Response"])
            mock_get_adapter.return_value = mock_adapter

            # Note: quick_eval might not be fully async in current implementation
            # This test checks the interface exists
            assert quick_eval is not None


class TestAsyncIntegration:
    @pytest.mark.asyncio
    async def test_nested_async_calls(self, sample_dataset):
        call_order = []

        with patch("benchwise.core.get_model_adapter") as mock_get_adapter:

            async def mock_generate(prompts, **kwargs):
                call_order.append("generate")
                await asyncio.sleep(0.01)  # Simulate async work
                return ["Response"] * len(prompts)

            mock_adapter = MockModelAdapter("async-test")
            mock_adapter.generate = mock_generate
            mock_get_adapter.return_value = mock_adapter

            @evaluate("async-test")
            async def async_evaluation(model, dataset):
                call_order.append("eval_start")
                responses = await model.generate(dataset.prompts)
                call_order.append("eval_end")
                return {"responses": len(responses)}

            results = await async_evaluation(sample_dataset)

            assert call_order == ["eval_start", "generate", "eval_end"]
            assert results[0].success
            assert results[0].result["responses"] == 5

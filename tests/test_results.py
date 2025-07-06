"""
Tests for results functionality
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from benchwise.results import (
    EvaluationResult,
    BenchmarkResult,
    ResultsAnalyzer,
    ResultsCache,
    save_results,
    load_results,
    cache,
)


class TestEvaluationResult:
    """Test EvaluationResult class"""

    def test_evaluation_result_creation(self):
        result = EvaluationResult(
            model_name="test-model",
            test_name="test_evaluation",
            result={"accuracy": 0.8},
            duration=1.5,
        )

        assert result.model_name == "test-model"
        assert result.test_name == "test_evaluation"
        assert result.result == {"accuracy": 0.8}
        assert result.duration == 1.5
        assert result.success
        assert not result.failed
        assert isinstance(result.timestamp, datetime)

    def test_evaluation_result_with_error(self):
        result = EvaluationResult(
            model_name="failing-model",
            test_name="test_evaluation",
            error="Model failed to respond",
            duration=0.0,
        )

        assert not result.success
        assert result.failed
        assert result.error == "Model failed to respond"
        assert result.result is None

    def test_evaluation_result_to_dict(self):
        result = EvaluationResult(
            model_name="test-model",
            test_name="test_evaluation",
            result={"accuracy": 0.8},
            duration=1.5,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["model_name"] == "test-model"
        assert result_dict["success"]
        assert "timestamp" in result_dict

    def test_get_score(self):
        result = EvaluationResult(
            model_name="test-model",
            test_name="test",
            result={"accuracy": 0.8, "f1": 0.75},
        )

        assert result.get_score("accuracy") == 0.8
        assert result.get_score("f1") == 0.75
        assert result.get_score("nonexistent") is None


class TestBenchmarkResult:
    """Test BenchmarkResult class"""

    def test_benchmark_result_creation(self):
        result = BenchmarkResult(
            benchmark_name="test_benchmark", metadata={"description": "Test benchmark"}
        )

        assert result.benchmark_name == "test_benchmark"
        assert result.metadata["description"] == "Test benchmark"
        assert len(result.results) == 0
        assert isinstance(result.timestamp, datetime)

    def test_add_result(self, sample_evaluation_result):
        benchmark = BenchmarkResult("test_benchmark")

        benchmark.add_result(sample_evaluation_result)

        assert len(benchmark.results) == 1
        assert benchmark.results[0] == sample_evaluation_result

    def test_model_names_property(self, sample_benchmark_result):
        model_names = sample_benchmark_result.model_names

        assert isinstance(model_names, list)
        assert "test-model" in model_names
        assert "failed-model" in model_names
        assert len(model_names) == 2

    def test_successful_results_property(self, sample_benchmark_result):
        successful = sample_benchmark_result.successful_results

        assert len(successful) == 1
        assert successful[0].model_name == "test-model"
        assert successful[0].success

    def test_success_rate_property(self, sample_benchmark_result):
        success_rate = sample_benchmark_result.success_rate

        assert success_rate == 0.5  # 1 success out of 2 total

    def test_to_dict(self, sample_benchmark_result):
        result_dict = sample_benchmark_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["benchmark_name"] == "test_benchmark"
        assert "results" in result_dict
        assert "summary" in result_dict


class TestResultsAnalyzer:
    """Basic tests for ResultsAnalyzer"""

    def test_analyze_model_performance_basic(self):
        results = [
            EvaluationResult("model-a", "test1", {"accuracy": 0.8}, 1.0),
            EvaluationResult("model-a", "test2", {"accuracy": 0.9}, 1.2),
        ]

        analysis = ResultsAnalyzer.analyze_model_performance(results, "accuracy")

        assert isinstance(analysis, dict)
        assert "model_name" in analysis
        assert "mean_score" in analysis
        assert analysis["model_name"] == "model-a"
        assert analysis["total_evaluations"] == 2

    def test_generate_text_report(self, sample_benchmark_result):
        report = ResultsAnalyzer.generate_report(sample_benchmark_result, "text")

        assert isinstance(report, str)
        assert "test_benchmark" in report
        assert "test-model" in report

    def test_generate_markdown_report(self, sample_benchmark_result):
        report = ResultsAnalyzer.generate_report(sample_benchmark_result, "markdown")

        assert isinstance(report, str)
        assert "# Benchmark Report" in report
        assert "test_benchmark" in report


class TestResultsCache:
    """Basic tests for ResultsCache"""

    def test_cache_creation(self, temp_cache_dir):
        cache = ResultsCache(temp_cache_dir)

        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.cache_dir.exists()

    def test_cache_save_and_load(self, temp_cache_dir, sample_evaluation_result):
        cache_instance = ResultsCache(temp_cache_dir)

        # Save result
        cache_instance.save_result(sample_evaluation_result, "test_hash")

        # Load result
        loaded = cache_instance.load_result(
            sample_evaluation_result.model_name,
            sample_evaluation_result.test_name,
            "test_hash",
        )

        cache_files = list(Path(temp_cache_dir).glob("*.json"))
        assert len(cache_files) > 0, f"No cache files created in {temp_cache_dir}"

        assert (
            loaded is not None
        ), f"Failed to load cached result. Cache files: {cache_files}"
        assert loaded.model_name == sample_evaluation_result.model_name
        assert loaded.test_name == sample_evaluation_result.test_name

    def test_cache_clear(self, temp_cache_dir, sample_evaluation_result):
        cache = ResultsCache(temp_cache_dir)

        # Save result
        cache.save_result(sample_evaluation_result, "test_hash")

        # Clear cache
        cache.clear_cache()

        # Try to load - should return None
        loaded = cache.load_result(
            sample_evaluation_result.model_name,
            sample_evaluation_result.test_name,
            "test_hash",
        )

        assert loaded is None


class TestSaveLoadResults:
    def test_save_and_load_json(self, sample_benchmark_result):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save results
            save_results(sample_benchmark_result, temp_path, "json")

            # Load results
            loaded = load_results(temp_path)

            assert loaded.benchmark_name == sample_benchmark_result.benchmark_name
            assert len(loaded.results) == len(sample_benchmark_result.results)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_csv(self, sample_benchmark_result):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            # Should not raise exception
            save_results(sample_benchmark_result, temp_path, "csv")

            # Check file exists
            assert Path(temp_path).exists()

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_markdown(self, sample_benchmark_result):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = f.name

        try:
            # Should not raise exception
            save_results(sample_benchmark_result, temp_path, "markdown")

            # Check file exists and has content
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert "test_benchmark" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_invalid_format(self, sample_benchmark_result):
        with pytest.raises(ValueError, match="Unsupported format"):
            save_results(sample_benchmark_result, "test.txt", "invalid")


class TestGlobalCache:
    def test_global_cache_exists(self):
        assert cache is not None
        assert hasattr(cache, "save_result")
        assert hasattr(cache, "load_result")
        assert hasattr(cache, "clear_cache")


class TestResultsEdgeCases:
    def test_benchmark_result_empty(self):
        benchmark = BenchmarkResult("empty_benchmark")

        assert benchmark.success_rate == 0.0
        assert len(benchmark.successful_results) == 0
        assert len(benchmark.failed_results) == 0
        assert benchmark.get_best_model() is None

    def test_evaluation_result_none_values(self):
        result = EvaluationResult(
            model_name="test", test_name="test", result=None, duration=0.0
        )

        assert result.get_score() is None
        assert result.get_score("anything") is None

    def test_results_analyzer_empty_results(self):
        analysis = ResultsAnalyzer.analyze_model_performance([], "accuracy")

        assert "error" in analysis
        assert "No results provided" in analysis["error"]

    def test_cache_nonexistent_result(self, temp_cache_dir):
        cache = ResultsCache(temp_cache_dir)

        result = cache.load_result("nonexistent", "test", "hash")

        assert result is None

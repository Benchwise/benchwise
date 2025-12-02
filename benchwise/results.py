from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import numpy as np
import hashlib


@dataclass
class EvaluationResult:
    """
    Result of a single model evaluation.

    Attributes:
        model_name: Name of the evaluated model
        test_name: Name of the test/benchmark
        result: The evaluation result (scores, metrics, etc.)
        duration: Time taken for evaluation in seconds
        dataset_info: Information about the dataset used
        error: Error message if evaluation failed
        metadata: Additional metadata about the evaluation
        timestamp: When the evaluation was completed
    """

    model_name: str
    test_name: str
    result: Any = None
    duration: float = 0.0
    dataset_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Whether the evaluation completed successfully."""
        return self.error is None

    @property
    def failed(self) -> bool:
        """Whether the evaluation failed."""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "model_name": self.model_name,
            "test_name": self.test_name,
            "result": self.result,
            "duration": self.duration,
            "dataset_info": self.dataset_info,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }

    def get_score(self, metric_name: Optional[str] = None) -> Union[float, Any]:
        """
        Extract a specific score from the result.

        Args:
            metric_name: Name of the metric to extract. If None, returns the main result.

        Returns:
            The score value
        """
        if metric_name is None:
            return self.result

        if isinstance(self.result, dict):
            return self.result.get(metric_name)

        return None


@dataclass
class BenchmarkResult:
    """
    Results from running a benchmark across multiple models.

    Attributes:
        benchmark_name: Name of the benchmark
        results: List of individual evaluation results
        metadata: Additional metadata about the benchmark run
        timestamp: When the benchmark was completed
    """

    benchmark_name: str
    results: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result to the benchmark."""
        self.results.append(result)

    @property
    def model_names(self) -> List[str]:
        """Get list of model names that were evaluated."""
        return [result.model_name for result in self.results]

    @property
    def successful_results(self) -> List[EvaluationResult]:
        """Get only successful evaluation results."""
        return [result for result in self.results if result.success]

    @property
    def failed_results(self) -> List[EvaluationResult]:
        """Get only failed evaluation results."""
        return [result for result in self.results if result.failed]

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of evaluations."""
        if not self.results:
            return 0.0
        return len(self.successful_results) / len(self.results)

    def get_best_model(
        self, metric_name: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """
        Get the best performing model result.

        Args:
            metric_name: Specific metric to compare. If None, compares the main result.

        Returns:
            EvaluationResult of the best performing model
        """
        successful_results = self.successful_results
        if not successful_results:
            return None

        return max(successful_results, key=lambda r: r.get_score(metric_name) or 0)

    def get_worst_model(
        self, metric_name: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """
        Get the worst performing model result.

        Args:
            metric_name: Specific metric to compare. If None, compares the main result.

        Returns:
            EvaluationResult of the worst performing model
        """
        successful_results = self.successful_results
        if not successful_results:
            return None

        return min(
            successful_results, key=lambda r: r.get_score(metric_name) or float("inf")
        )

    def compare_models(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare all models in the benchmark.

        Args:
            metric_name: Specific metric to compare

        Returns:
            Dictionary with comparison statistics
        """
        successful_results = self.successful_results
        if not successful_results:
            return {"error": "No successful results to compare"}

        scores = [result.get_score(metric_name) for result in successful_results]
        model_names = [result.model_name for result in successful_results]

        # Filter out None scores
        valid_scores = [
            (name, score)
            for name, score in zip(model_names, scores)
            if score is not None
        ]

        if not valid_scores:
            return {"error": "No valid scores found"}

        sorted_results = sorted(valid_scores, key=lambda x: x[1], reverse=True)

        return {
            "ranking": [
                {"model": name, "score": score} for name, score in sorted_results
            ],
            "best_model": sorted_results[0][0],
            "best_score": sorted_results[0][1],
            "worst_model": sorted_results[-1][0],
            "worst_score": sorted_results[-1][1],
            "mean_score": np.mean([score for _, score in valid_scores]),
            "std_score": np.std([score for _, score in valid_scores]),
            "total_models": len(valid_scores),
        }

    def get_model_result(self, model_name: str) -> Optional[EvaluationResult]:
        """Get result for a specific model."""
        for result in self.results:
            if result.model_name == model_name:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary format."""
        return {
            "benchmark_name": self.benchmark_name,
            "results": [result.to_dict() for result in self.results],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_models": len(self.results),
                "successful_models": len(self.successful_results),
                "failed_models": len(self.failed_results),
                "success_rate": self.success_rate,
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        data = []
        for result in self.results:
            row = {
                "model_name": result.model_name,
                "test_name": result.test_name,
                "duration": result.duration,
                "success": result.success,
                "error": result.error,
                "timestamp": result.timestamp,
            }

            # Flatten result metrics
            if isinstance(result.result, dict):
                for key, value in result.result.items():
                    row[f"metric_{key}"] = value
            else:
                row["result"] = result.result

            data.append(row)

        return pd.DataFrame(data)

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Save benchmark results to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_to_csv(self, file_path: Union[str, Path]) -> None:
        """Save benchmark results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)


class ResultsAnalyzer:
    """Utility class for analyzing evaluation results."""

    @staticmethod
    def compare_benchmarks(
        benchmark_results: List[BenchmarkResult], metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare results across multiple benchmarks.

        Args:
            benchmark_results: List of benchmark results to compare
            metric_name: Specific metric to compare

        Returns:
            Dictionary with cross-benchmark comparison
        """
        comparison: Dict[str, Any] = {
            "benchmarks": [],
            "models": set(),
            "cross_benchmark_scores": {},
        }

        for benchmark in benchmark_results:
            benchmark_info = {
                "name": benchmark.benchmark_name,
                "timestamp": benchmark.timestamp,
                "models": benchmark.model_names,
                "success_rate": benchmark.success_rate,
            }

            comparison["benchmarks"].append(benchmark_info)
            comparison["models"].update(benchmark.model_names)

            # Collect scores for each model
            for result in benchmark.successful_results:
                model_name = result.model_name
                score = result.get_score(metric_name)

                if model_name not in comparison["cross_benchmark_scores"]:
                    comparison["cross_benchmark_scores"][model_name] = {}

                comparison["cross_benchmark_scores"][model_name][
                    benchmark.benchmark_name
                ] = score

        comparison["models"] = list(comparison["models"])

        return comparison

    @staticmethod
    def analyze_model_performance(
        results: List[EvaluationResult], metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance of a single model across multiple evaluations.

        Args:
            results: List of evaluation results for the same model
            metric_name: Specific metric to analyze

        Returns:
            Dictionary with performance analysis
        """
        if not results:
            return {"error": "No results provided"}

        model_name = results[0].model_name
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"error": "No successful results found"}

        scores = [result.get_score(metric_name) for result in successful_results]
        valid_scores = [score for score in scores if score is not None]

        if not valid_scores:
            return {"error": "No valid scores found"}

        return {
            "model_name": model_name,
            "total_evaluations": len(results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "mean_score": np.mean(valid_scores),
            "median_score": np.median(valid_scores),
            "std_score": np.std(valid_scores),
            "min_score": np.min(valid_scores),
            "max_score": np.max(valid_scores),
            "score_range": np.max(valid_scores) - np.min(valid_scores),
        }

    @staticmethod
    def generate_report(
        benchmark_result: BenchmarkResult, output_format: str = "text"
    ) -> str:
        """
        Generate a formatted report of benchmark results.

        Args:
            benchmark_result: Benchmark result to report on
            output_format: Format of the report ('text', 'markdown', 'html')

        Returns:
            Formatted report string
        """
        if output_format == "text":
            return ResultsAnalyzer._generate_text_report(benchmark_result)
        elif output_format == "markdown":
            return ResultsAnalyzer._generate_markdown_report(benchmark_result)
        elif output_format == "html":
            return ResultsAnalyzer._generate_html_report(benchmark_result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    @staticmethod
    def _generate_text_report(benchmark_result: BenchmarkResult) -> str:
        """Generate plain text report."""
        lines = [
            f"Benchmark Report: {benchmark_result.benchmark_name}",
            "=" * 50,
            f"Timestamp: {benchmark_result.timestamp}",
            f"Total Models: {len(benchmark_result.results)}",
            f"Successful: {len(benchmark_result.successful_results)}",
            f"Failed: {len(benchmark_result.failed_results)}",
            f"Success Rate: {benchmark_result.success_rate:.2%}",
            "",
            "Results:",
        ]

        for result in benchmark_result.results:
            status = "✓" if result.success else "✗"
            lines.append(f"  {status} {result.model_name}: {result.result}")
            if result.error:
                lines.append(f"    Error: {result.error}")

        return "\n".join(lines)

    @staticmethod
    def _generate_markdown_report(benchmark_result: BenchmarkResult) -> str:
        """Generate markdown report."""
        lines = [
            f"# Benchmark Report: {benchmark_result.benchmark_name}",
            "",
            f"**Timestamp:** {benchmark_result.timestamp}  ",
            f"**Total Models:** {len(benchmark_result.results)}  ",
            f"**Successful:** {len(benchmark_result.successful_results)}  ",
            f"**Failed:** {len(benchmark_result.failed_results)}  ",
            f"**Success Rate:** {benchmark_result.success_rate:.2%}  ",
            "",
            "## Results",
            "",
            "| Model | Status | Result | Duration |",
            "|-------|--------|--------|----------|",
        ]

        for result in benchmark_result.results:
            status = "✅ Success" if result.success else "❌ Failed"
            result_str = str(result.result) if result.success else result.error
            lines.append(
                f"| {result.model_name} | {status} | {result_str} | {result.duration:.2f}s |"
            )

        return "\n".join(lines)

    @staticmethod
    def _generate_html_report(benchmark_result: BenchmarkResult) -> str:
        """Generate HTML report."""
        html = f"""
        <html>
        <head>
            <title>Benchmark Report: {benchmark_result.benchmark_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report: {benchmark_result.benchmark_name}</h1>
            <p><strong>Timestamp:</strong> {benchmark_result.timestamp}</p>
            <p><strong>Total Models:</strong> {len(benchmark_result.results)}</p>
            <p><strong>Success Rate:</strong> {benchmark_result.success_rate:.2%}</p>

            <h2>Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Result</th>
                    <th>Duration</th>
                </tr>
        """

        for result in benchmark_result.results:
            status_class = "success" if result.success else "failed"
            status_text = "Success" if result.success else "Failed"
            result_text = str(result.result) if result.success else result.error

            html += f"""
                <tr>
                    <td>{result.model_name}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result_text}</td>
                    <td>{result.duration:.2f}s</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html


class ResultsCache:
    """Cache for storing and retrieving evaluation results."""

    def __init__(self, cache_dir: Union[str, Path] = "benchmark_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, model_name: str, test_name: str, dataset_hash: str) -> str:
        """Generate cache key for a result."""
        key_data = f"{model_name}_{test_name}_{dataset_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def save_result(self, result: EvaluationResult, dataset_hash: str) -> None:
        """Save evaluation result to cache."""
        cache_key = self._get_cache_key(
            result.model_name, result.test_name, dataset_hash
        )
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def load_result(
        self, model_name: str, test_name: str, dataset_hash: str
    ) -> Optional[EvaluationResult]:
        """Load evaluation result from cache."""
        cache_key = self._get_cache_key(model_name, test_name, dataset_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            if "timestamp" in data:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])

            # Remove computed properties that shouldn't be passed to constructor
            data_clean = {
                k: v for k, v in data.items() if k not in ["success", "failed"]
            }  # These are @property methods

            return EvaluationResult(**data_clean)
        except Exception:
            return None

    def clear_cache(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def list_cached_results(self) -> List[Dict[str, Any]]:
        """List all cached results."""
        results = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                results.append(
                    {
                        "file": cache_file.name,
                        "model_name": data.get("model_name"),
                        "test_name": data.get("test_name"),
                        "timestamp": data.get("timestamp"),
                    }
                )
            except Exception:
                continue
        return results


cache = ResultsCache()


def save_results(
    benchmark_result: BenchmarkResult, file_path: Union[str, Path], format: str = "json"
) -> None:
    """
    Save benchmark results to file.

    Args:
        benchmark_result: Benchmark result to save
        file_path: Path to save the file
        format: File format ('json', 'csv', 'markdown', 'html')
    """
    if format == "json":
        benchmark_result.save_to_json(file_path)
    elif format == "csv":
        benchmark_result.save_to_csv(file_path)
    elif format == "markdown":
        report = ResultsAnalyzer.generate_report(benchmark_result, "markdown")
        with open(file_path, "w") as f:
            f.write(report)
    elif format == "html":
        report = ResultsAnalyzer.generate_report(benchmark_result, "html")
        with open(file_path, "w") as f:
            f.write(report)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(file_path: Union[str, Path]) -> BenchmarkResult:
    """
    Load benchmark results from file.

    Args:
        file_path: Path to the results file

    Returns:
        BenchmarkResult object
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if "timestamp" in data:
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

    results = []
    for result_data in data.get("results", []):
        if "timestamp" in result_data:
            result_data["timestamp"] = datetime.fromisoformat(result_data["timestamp"])

        result_data_clean = {
            k: v for k, v in result_data.items() if k not in ["success", "failed"]
        }

        results.append(EvaluationResult(**result_data_clean))

    return BenchmarkResult(
        benchmark_name=data["benchmark_name"],
        results=results,
        metadata=data.get("metadata", {}),
        timestamp=data["timestamp"],
    )

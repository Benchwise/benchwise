from typing import List, Dict, Any, Callable, Optional
from functools import wraps
import asyncio
import time
from .models import get_model_adapter
from .datasets import Dataset
from .results import EvaluationResult
from .config import get_api_config
from .client import upload_results


def evaluate(*models: str, upload: bool = None, **kwargs) -> Callable:
    """
    Decorator for creating LLM evaluations.

    Args:
        *models: Model names to evaluate
        upload: Whether to upload results to BenchWise API (None = use config default)
        **kwargs: Additional evaluation parameters

    Usage:
        @evaluate("gpt-4", "claude-3")
        def test_summarization(model, dataset):
            responses = model.generate(dataset.prompts)
            scores = rouge_l(responses, dataset.ground_truth)
            assert scores.mean() > 0.6
            return scores

        # With auto-upload to BenchWise
        @evaluate("gpt-4", "claude-3", upload=True)
        def test_qa(model, dataset):
            # evaluation code
            return scores
    """

    def decorator(test_func: Callable) -> Callable:
        @wraps(test_func)
        async def wrapper(dataset: Dataset, **test_kwargs) -> List[EvaluationResult]:
            results = []

            for model_name in models:
                try:
                    # Get model adapter
                    model = get_model_adapter(model_name)

                    # Run the test
                    start_time = time.time()
                    result = await test_func(model, dataset, **test_kwargs)
                    end_time = time.time()

                    # Create evaluation result
                    # Combine decorator kwargs with benchmark metadata if available
                    combined_metadata = kwargs.copy()
                    # Check for metadata on both the original function and the wrapper
                    if hasattr(test_func, "_benchmark_metadata"):
                        combined_metadata.update(test_func._benchmark_metadata)
                    elif hasattr(wrapper, "_benchmark_metadata"):
                        combined_metadata.update(wrapper._benchmark_metadata)

                    eval_result = EvaluationResult(
                        model_name=model_name,
                        test_name=test_func.__name__,
                        result=result,
                        duration=end_time - start_time,
                        dataset_info=dataset.metadata,
                        metadata=combined_metadata,
                    )
                    results.append(eval_result)

                except Exception as e:
                    # Handle errors gracefully
                    # Combine decorator kwargs with benchmark metadata if available
                    combined_metadata = kwargs.copy()
                    # Check for metadata on both the original function and the wrapper
                    if hasattr(test_func, "_benchmark_metadata"):
                        combined_metadata.update(test_func._benchmark_metadata)
                    elif hasattr(wrapper, "_benchmark_metadata"):
                        combined_metadata.update(wrapper._benchmark_metadata)

                    eval_result = EvaluationResult(
                        model_name=model_name,
                        test_name=test_func.__name__,
                        error=str(e),
                        duration=0,
                        dataset_info=dataset.metadata,
                        metadata=combined_metadata,
                    )
                    results.append(eval_result)

            # Upload results to BenchWise API if enabled
            config = get_api_config()
            should_upload = upload if upload is not None else config.upload_enabled

            if should_upload and results:
                try:
                    await upload_results(
                        results, test_func.__name__, dataset.metadata or {}
                    )
                except Exception as e:
                    if config.debug:
                        print(f"⚠️ Upload failed (results saved locally): {e}")

            return results

        # Preserve any existing metadata from the original function
        if hasattr(test_func, "_benchmark_metadata"):
            wrapper._benchmark_metadata = test_func._benchmark_metadata

        return wrapper

    return decorator


def benchmark(name: str, description: str = "", **kwargs) -> Callable:
    """
    Decorator for creating benchmarks.

    Usage:
        @benchmark("medical_qa", "Medical question answering benchmark")
        def medical_qa_test(model, dataset):
            # Test implementation
            pass
    """

    def decorator(test_func: Callable) -> Callable:
        # Add metadata to function before wrapping
        test_func._benchmark_metadata = {
            "name": name,
            "description": description,
            **kwargs,
        }

        # Return the function with metadata attached
        return test_func

    return decorator


def stress_test(concurrent_requests: int = 10, duration: int = 60) -> Callable:
    """
    Decorator for stress testing LLMs.

    Usage:
        @stress_test(concurrent_requests=50, duration=120)
        def load_test(model, dataset):
            # Test implementation
            pass
    """

    def decorator(test_func: Callable) -> Callable:
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            # Implementation for stress testing
            tasks = []
            start_time = time.time()

            while time.time() - start_time < duration:
                # Create concurrent tasks
                batch_tasks = [
                    test_func(*args, **kwargs) for _ in range(concurrent_requests)
                ]

                # Run batch
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                tasks.extend(batch_results)

                # Small delay between batches
                await asyncio.sleep(0.1)

            return tasks

        return wrapper

    return decorator


class EvaluationRunner:
    """Main class for running evaluations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results_cache = {}

    async def run_evaluation(
        self, test_func: Callable, dataset: Dataset, models: List[str]
    ) -> List[EvaluationResult]:
        """Run evaluation on multiple models."""
        results = []

        for model_name in models:
            model = get_model_adapter(model_name)
            result = await test_func(model, dataset)
            results.append(result)

        return results

    def compare_models(
        self, results: List[EvaluationResult], metric_name: str = None
    ) -> Dict[str, Any]:
        """Compare model performance."""
        # Filter successful results
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"error": "No successful results to compare"}

        # Extract scores for comparison
        model_scores = []
        for r in successful_results:
            if metric_name:
                score = r.get_score(metric_name)
            else:
                # If result is a dict, try to find a numeric value
                if isinstance(r.result, dict):
                    # Look for common metric names
                    for key in ["accuracy", "f1", "score", "rouge_l_f1"]:
                        if key in r.result and isinstance(r.result[key], (int, float)):
                            score = r.result[key]
                            break
                    else:
                        # Take the first numeric value found
                        for value in r.result.values():
                            if isinstance(value, (int, float)):
                                score = value
                                break
                        else:
                            score = 0  # Default if no numeric value found
                else:
                    score = r.result if isinstance(r.result, (int, float)) else 0

            model_scores.append((r.model_name, score if score is not None else 0))

        if not model_scores:
            return {"error": "No comparable scores found"}

        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        comparison = {
            "models": [r.model_name for r in successful_results],
            "scores": [score for _, score in model_scores],
            "best_model": model_scores[0][0],
            "worst_model": model_scores[-1][0],
            "ranking": [
                {"model": name, "score": score} for name, score in model_scores
            ],
        }
        return comparison


# Convenience functions
def run_benchmark(
    benchmark_func: Callable, dataset: Dataset, models: List[str]
) -> List[EvaluationResult]:
    """Run a benchmark on multiple models."""
    runner = EvaluationRunner()
    return asyncio.run(runner.run_evaluation(benchmark_func, dataset, models))


def quick_eval(prompt: str, models: List[str], metric: Callable) -> Dict[str, float]:
    """Quick evaluation with a single prompt."""
    results = {}

    for model_name in models:
        model = get_model_adapter(model_name)
        response = model.generate([prompt])[0]
        score = metric(response)
        results[model_name] = score

    return results

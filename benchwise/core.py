from typing import List, Dict, Any, Callable, Optional, Union
from functools import wraps
import asyncio
import time
import inspect
import logging
from .models import get_model_adapter
from .datasets import Dataset
from .results import EvaluationResult
from .config import get_api_config
from .client import upload_results

# Set up logger
logger = logging.getLogger("benchwise")


def evaluate(*models: str, upload: bool = None, **kwargs) -> Callable:
    """
    Decorator for creating LLM evaluations.
    
    Supports both sync and async test functions automatically.

    Args:
        *models: Model names to evaluate
        upload: Whether to upload results to Benchwise API (None = use config default)
        **kwargs: Additional evaluation parameters

    Usage:
        # Sync version
        @evaluate("gpt-4", "claude-3")
        def test_summarization(model, dataset):
            responses = asyncio.run(model.generate(dataset.prompts))
            scores = rouge_l(responses, dataset.ground_truth)
            return scores

        # Async version (recommended)
        @evaluate("gpt-4", "claude-3")
        async def test_summarization(model, dataset):
            responses = await model.generate(dataset.prompts)
            scores = rouge_l(responses, dataset.ground_truth)
            return scores

        # With auto-upload to Benchwise
        @evaluate("gpt-4", "claude-3", upload=True)
        async def test_qa(model, dataset):
            # evaluation code
            return scores
    """

    def decorator(test_func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(test_func)
        
        if is_async:
            @wraps(test_func)
            async def async_wrapper(dataset: Dataset, **test_kwargs) -> List[EvaluationResult]:
                return await _run_evaluation(test_func, dataset, models, upload, kwargs, test_kwargs, is_async=True)
            
            # Preserve metadata
            if hasattr(test_func, "_benchmark_metadata"):
                async_wrapper._benchmark_metadata = test_func._benchmark_metadata
            
            return async_wrapper
        else:
            @wraps(test_func)
            def sync_wrapper(dataset: Dataset, **test_kwargs) -> List[EvaluationResult]:
                return asyncio.run(_run_evaluation(test_func, dataset, models, upload, kwargs, test_kwargs, is_async=False))
            
            # Preserve metadata
            if hasattr(test_func, "_benchmark_metadata"):
                sync_wrapper._benchmark_metadata = test_func._benchmark_metadata
            
            return sync_wrapper

    return decorator


async def _run_evaluation(
    test_func: Callable,
    dataset: Dataset,
    models: tuple,
    upload: Optional[bool],
    decorator_kwargs: Dict[str, Any],
    test_kwargs: Dict[str, Any],
    is_async: bool
) -> List[EvaluationResult]:
    """Internal function to run evaluation logic."""
    results = []
    
    logger.info(f"Starting evaluation: {test_func.__name__} on {len(models)} model(s)")

    for model_name in models:
        try:
            logger.debug(f"Evaluating model: {model_name}")
            
            # Get model adapter
            model = get_model_adapter(model_name)

            # Run the test
            start_time = time.time()
            
            if is_async:
                result = await test_func(model, dataset, **test_kwargs)
            else:
                result = test_func(model, dataset, **test_kwargs)
            
            end_time = time.time()

            # Combine decorator kwargs with benchmark metadata if available
            combined_metadata = decorator_kwargs.copy()
            if hasattr(test_func, "_benchmark_metadata"):
                combined_metadata.update(test_func._benchmark_metadata)

            eval_result = EvaluationResult(
                model_name=model_name,
                test_name=test_func.__name__,
                result=result,
                duration=end_time - start_time,
                dataset_info=dataset.metadata,
                metadata=combined_metadata,
            )
            results.append(eval_result)
            
            logger.info(f"✓ {model_name} completed in {end_time - start_time:.2f}s")

        except Exception as e:
            logger.error(f"✗ {model_name} failed: {e}", exc_info=True)
            
            # Combine decorator kwargs with benchmark metadata if available
            combined_metadata = decorator_kwargs.copy()
            if hasattr(test_func, "_benchmark_metadata"):
                combined_metadata.update(test_func._benchmark_metadata)

            eval_result = EvaluationResult(
                model_name=model_name,
                test_name=test_func.__name__,
                error=str(e),
                duration=0,
                dataset_info=dataset.metadata,
                metadata=combined_metadata,
            )
            results.append(eval_result)

    # Upload results to Benchwise API if enabled
    config = get_api_config()
    should_upload = upload if upload is not None else config.upload_enabled

    if should_upload and results:
        try:
            logger.debug("Uploading results to Benchwise API")
            await upload_results(
                results, test_func.__name__, dataset.metadata or {}
            )
            logger.info("Results uploaded successfully")
        except Exception as e:
            logger.warning(f"Upload failed (results saved locally): {e}")
            if config.debug:
                logger.debug("Upload error details", exc_info=True)

    return results


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
    
    NOTE: This is a WIP feature and may not be fully functional yet.

    Usage:
        @stress_test(concurrent_requests=50, duration=120)
        async def load_test(model, dataset):
            # Test implementation
            pass
    """

    def decorator(test_func: Callable) -> Callable:
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            logger.info(f"Starting stress test: {concurrent_requests} concurrent requests for {duration}s")
            
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

            logger.info(f"Stress test completed: {len(tasks)} total requests")
            return tasks

        return wrapper

    return decorator


class EvaluationRunner:
    """Main class for running evaluations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results_cache = {}
        self.logger = logging.getLogger("benchwise.runner")

    async def run_evaluation(
        self, test_func: Callable, dataset: Dataset, models: List[str]
    ) -> List[EvaluationResult]:
        """Run evaluation on multiple models."""
        results = []
        
        self.logger.info(f"Running evaluation on {len(models)} models")

        for model_name in models:
            try:
                model = get_model_adapter(model_name)
                result = await test_func(model, dataset)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Evaluation failed for {model_name}: {e}")

        return results

    def compare_models(
        self, results: List[EvaluationResult], metric_name: str = None
    ) -> Dict[str, Any]:
        """Compare model performance."""
        # Filter successful results
        successful_results = [r for r in results if r.success]

        if not successful_results:
            self.logger.warning("No successful results to compare")
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
        
        self.logger.info(f"Comparison complete: Best model is {comparison['best_model']}")
        
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
    
    logger.info(f"Running quick eval on {len(models)} models")

    for model_name in models:
        try:
            model = get_model_adapter(model_name)
            response = asyncio.run(model.generate([prompt]))[0]
            score = metric(response)
            results[model_name] = score
        except Exception as e:
            logger.error(f"Quick eval failed for {model_name}: {e}")
            results[model_name] = None

    return results

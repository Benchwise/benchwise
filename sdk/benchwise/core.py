from typing import List, Dict, Any, Callable, Optional
from functools import wraps
import asyncio
import time
from .models import ModelAdapter, get_model_adapter
from .datasets import Dataset
from .results import EvaluationResult


def evaluate(*models: str, **kwargs) -> Callable:
    """
    Decorator for creating LLM evaluations.
    
    Usage:
        @evaluate("gpt-4", "claude-3")
        def test_summarization(model, dataset):
            responses = model.generate(dataset.prompts)
            scores = rouge_l(responses, dataset.ground_truth)
            assert scores.mean() > 0.6
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
                    eval_result = EvaluationResult(
                        model_name=model_name,
                        test_name=test_func.__name__,
                        result=result,
                        duration=end_time - start_time,
                        dataset_info=dataset.metadata,
                        **kwargs
                    )
                    results.append(eval_result)
                    
                except Exception as e:
                    eval_result = EvaluationResult(
                        model_name=model_name,
                        test_name=test_func.__name__,
                        error=str(e),
                        duration=0,
                        dataset_info=dataset.metadata,
                        **kwargs
                    )
                    results.append(eval_result)
            
            return results
        
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
        @wraps(test_func)
        async def wrapper(*args, **test_kwargs):
            # Add benchmark metadata
            test_kwargs.update({
                'benchmark_name': name,
                'description': description,
                **kwargs
            })
            
            return await test_func(*args, **test_kwargs)
        
        # Add metadata to function
        wrapper._benchmark_metadata = {
            'name': name,
            'description': description,
            **kwargs
        }
        
        return wrapper
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
            tasks = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Create concurrent tasks
                batch_tasks = [
                    test_func(*args, **kwargs) 
                    for _ in range(concurrent_requests)
                ]
                
                # Run batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
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
    
    async def run_evaluation(self, test_func: Callable, dataset: Dataset, models: List[str]) -> List[EvaluationResult]:
        """Run evaluation on multiple models."""
        results = []
        
        for model_name in models:
            model = get_model_adapter(model_name)
            result = await test_func(model, dataset)
            results.append(result)
        
        return results
    
    def compare_models(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare model performance."""
        comparison = {
            'models': [r.model_name for r in results],
            'scores': [r.result for r in results],
            'best_model': max(results, key=lambda x: x.result).model_name,
            'worst_model': min(results, key=lambda x: x.result).model_name,
        }
        return comparison


# Convenience functions
def run_benchmark(benchmark_func: Callable, dataset: Dataset, models: List[str]) -> List[EvaluationResult]:
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

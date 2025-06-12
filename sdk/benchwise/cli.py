"""
BenchWise CLI - Command line interface for LLM evaluation
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from . import __version__
from .datasets import load_dataset
from .models import get_model_adapter
from .metrics import get_text_generation_metrics, get_qa_metrics
from .results import save_results, BenchmarkResult, EvaluationResult


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="benchwise",
        description="BenchWise CLI - The GitHub of LLM Evaluation"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"BenchWise {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Run evaluations")
    eval_parser.add_argument("models", nargs="+", help="Models to evaluate")
    eval_parser.add_argument("--dataset", "-d", required=True, help="Path to dataset file")
    eval_parser.add_argument("--metrics", "-m", nargs="+", default=["accuracy"], help="Metrics to compute")
    eval_parser.add_argument("--output", "-o", help="Output file path")
    eval_parser.add_argument("--format", "-f", choices=["json", "csv", "markdown"], default="json", help="Output format")
    eval_parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    eval_parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument("resource", choices=["models", "metrics", "datasets"], help="Resource type to list")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("dataset", help="Path to dataset file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare evaluation results")
    compare_parser.add_argument("results", nargs="+", help="Paths to result files")
    compare_parser.add_argument("--metric", "-m", help="Specific metric to compare")
    compare_parser.add_argument("--output", "-o", help="Output file path")
    
    return parser


async def run_evaluation(
    models: List[str],
    dataset_path: str,
    metrics: List[str],
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> BenchmarkResult:
    """Run evaluation on specified models."""
    
    try:
        dataset = load_dataset(dataset_path)
        print(f"Loaded dataset: {dataset.name} ({len(dataset.data)} items)")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    benchmark_result = BenchmarkResult(
        benchmark_name=f"cli_evaluation_{dataset.name}",
        metadata={
            "dataset_path": dataset_path,
            "models": models,
            "metrics": metrics,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    )
    
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        
        try:
            model = get_model_adapter(model_name)
            
            prompts = dataset.prompts
            if not prompts:
                print(f"Warning: No prompts found in dataset for {model_name}")
                continue
            
            print(f"Generating {len(prompts)} responses...")
            responses = await model.generate(
                prompts,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            references = dataset.references
            if not references:
                print(f"Warning: No references found in dataset for {model_name}")
                continue
            
            from .metrics import accuracy, rouge_l, semantic_similarity
            
            results = {}
            for metric_name in metrics:
                if metric_name == "accuracy":
                    metric_result = accuracy(responses, references)
                    results["accuracy"] = metric_result["accuracy"]
                elif metric_name == "rouge_l":
                    metric_result = rouge_l(responses, references)
                    results["rouge_l_f1"] = metric_result["f1"]
                elif metric_name == "semantic_similarity":
                    metric_result = semantic_similarity(responses, references)
                    results["semantic_similarity"] = metric_result["mean_similarity"]
                else:
                    print(f"Warning: Unknown metric '{metric_name}'")
            
            eval_result = EvaluationResult(
                model_name=model_name,
                test_name="cli_evaluation",
                result=results,
                dataset_info=dataset.metadata
            )
            
            benchmark_result.add_result(eval_result)
            print(f"✓ {model_name} completed: {results}")
            
        except Exception as e:
            eval_result = EvaluationResult(
                model_name=model_name,
                test_name="cli_evaluation",
                error=str(e),
                dataset_info=dataset.metadata
            )
            benchmark_result.add_result(eval_result)
            print(f"{model_name} failed: {e}")
    
    return benchmark_result


def list_resources(resource_type: str):
    """List available resources."""
    if resource_type == "models":
        print("Available model adapters:")
        print("  OpenAI: gpt-4, gpt-3.5-turbo, gpt-4o")
        print("  Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku")
        print("  Google: gemini-pro, gemini-1.5-pro")
        print("  HuggingFace: Any model ID from HuggingFace Hub")
    
    elif resource_type == "metrics":
        print("Available metrics:")
        print("  accuracy - Exact match accuracy")
        print("  rouge_l - ROUGE-L F1 score")
        print("  semantic_similarity - Semantic similarity using embeddings")
        print("  safety_score - Content safety evaluation")
        print("  coherence_score - Text coherence evaluation")
    
    elif resource_type == "datasets":
        print("Dataset format:")
        print("  JSON: List of objects with 'prompt'/'question' and 'answer'/'reference' fields")
        print("  CSV: Columns for prompts and references")
        print("  Example: [{'question': 'What is AI?', 'answer': 'Artificial Intelligence'}]")


def validate_dataset(dataset_path: str):
    """Validate dataset format."""
    try:
        dataset = load_dataset(dataset_path)
        print(f"✓ Dataset loaded successfully: {dataset.name}")
        print(f"  Size: {len(dataset.data)} items")
        
        if not dataset.prompts:
            print("⚠ Warning: No prompts found (looking for 'prompt', 'question', 'input', 'text' fields)")
        else:
            print(f"  Prompts: {len(dataset.prompts)} found")
        
        if not dataset.references:
            print("⚠ Warning: No references found (looking for 'reference', 'answer', 'output', 'target' fields)")
        else:
            print(f"  References: {len(dataset.references)} found")
        
        if dataset.schema:
            is_valid = dataset.validate_schema()
            print(f"  Schema validation: {'✓ Passed' if is_valid else '✗ Failed'}")
        
        stats = dataset.get_statistics()
        print(f"  Fields: {', '.join(stats['fields'])}")
        
        print("✓ Dataset validation completed")
        
    except Exception as e:
        print(f"✗ Dataset validation failed: {e}")
        sys.exit(1)


async def compare_results(result_paths: List[str], metric: Optional[str] = None):
    """Compare evaluation results."""
    from .results import load_results, ResultsAnalyzer
    
    try:
        benchmark_results = []
        for path in result_paths:
            result = load_results(path)
            benchmark_results.append(result)
            print(f"Loaded: {result.benchmark_name} ({len(result.results)} models)")
        
        comparison = ResultsAnalyzer.compare_benchmarks(benchmark_results, metric)
        
        print(f"\n=== Comparison Results ===")
        print(f"Benchmarks: {len(comparison['benchmarks'])}")
        print(f"Total models: {len(comparison['models'])}")
        
        for model_name in comparison['models']:
            scores = comparison['cross_benchmark_scores'].get(model_name, {})
            print(f"\n{model_name}:")
            for benchmark_name, score in scores.items():
                print(f"  {benchmark_name}: {score}")
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "eval":
        result = asyncio.run(run_evaluation(
            models=args.models,
            dataset_path=args.dataset,
            metrics=args.metrics,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        ))
        
        if args.output:
            save_results(result, args.output, args.format)
            print(f"\nResults saved to: {args.output}")
        else:
            print(f"\n=== Final Results ===")
            for eval_result in result.results:
                status = "✓" if eval_result.success else "✗"
                print(f"{status} {eval_result.model_name}: {eval_result.result}")
    
    elif args.command == "list":
        list_resources(args.resource)
    
    elif args.command == "validate":
        validate_dataset(args.dataset)
    
    elif args.command == "compare":
        asyncio.run(compare_results(args.results, args.metric))


if __name__ == "__main__":
    main()

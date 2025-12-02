"""
Benchwise CLI - Command line interface for LLM evaluation
"""

import argparse
import asyncio
import sys
from typing import List, Optional, Any, Dict

from . import __version__
from .datasets import load_dataset
from .models import get_model_adapter
from .results import save_results, BenchmarkResult, EvaluationResult
from .config import get_api_config, configure_benchwise
from .client import get_client, sync_offline_results


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="benchwise", description="Benchwise CLI - The GitHub of LLM Evaluation"
    )

    parser.add_argument(
        "--version", action="version", version=f"Benchwise {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Run evaluations")
    eval_parser.add_argument("models", nargs="+", help="Models to evaluate")
    eval_parser.add_argument(
        "--dataset", "-d", required=True, help="Path to dataset file"
    )
    eval_parser.add_argument(
        "--metrics", "-m", nargs="+", default=["accuracy"], help="Metrics to compute"
    )
    eval_parser.add_argument("--output", "-o", help="Output file path")
    eval_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format",
    )
    eval_parser.add_argument(
        "--temperature", type=float, default=0.0, help="Model temperature"
    )
    eval_parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Maximum tokens to generate"
    )
    eval_parser.add_argument(
        "--upload", action="store_true", help="Upload results to Benchwise API"
    )
    eval_parser.add_argument(
        "--no-upload", action="store_true", help="Disable result upload"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument(
        "resource",
        choices=["models", "metrics", "datasets"],
        help="Resource type to list",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("dataset", help="Path to dataset file")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare evaluation results")
    compare_parser.add_argument("results", nargs="+", help="Paths to result files")
    compare_parser.add_argument("--metric", "-m", help="Specific metric to compare")
    compare_parser.add_argument("--output", "-o", help="Output file path")

    # Configure command
    config_parser = subparsers.add_parser(
        "configure", help="Configure Benchwise settings"
    )
    config_parser.add_argument("--api-url", help="Benchwise API URL")
    config_parser.add_argument("--api-key", help="API authentication key")
    config_parser.add_argument(
        "--upload", choices=["true", "false"], help="Enable/disable automatic uploads"
    )
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )
    config_parser.add_argument(
        "--reset", action="store_true", help="Reset to default configuration"
    )

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync offline results with API")
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without uploading",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show Benchwise status")
    status_parser.add_argument(
        "--api", action="store_true", help="Check API connectivity"
    )
    status_parser.add_argument(
        "--auth", action="store_true", help="Check authentication status"
    )

    return parser


async def run_evaluation(
    models: List[str],
    dataset_path: str,
    metrics: List[str],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    upload: Optional[bool] = None,
) -> BenchmarkResult:
    """Run evaluation on specified models."""

    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
        print(f"Loaded dataset: {dataset.name} ({len(dataset.data)} items)")
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.")
        print("Please check the file path and ensure the file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset from '{dataset_path}': {e}")
        print("Supported formats: .json, .csv")
        sys.exit(1)

    # Create benchmark result
    benchmark_result = BenchmarkResult(
        benchmark_name=f"cli_evaluation_{dataset.name}",
        metadata={
            "dataset_path": dataset_path,
            "models": models,
            "metrics": metrics,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )

    # Run evaluation for each model
    for model_name in models:
        print(f"\nEvaluating {model_name}...")

        try:
            # Get model adapter
            model = get_model_adapter(model_name)

            # Check for API key requirements for cloud models
            if model_name.startswith(("gpt-", "claude-", "gemini-")):
                import os

                api_key_map = {
                    "gpt-": "OPENAI_API_KEY",
                    "claude-": "ANTHROPIC_API_KEY",
                    "gemini-": "GOOGLE_API_KEY",
                }

                for prefix, env_var in api_key_map.items():
                    if model_name.startswith(prefix) and not os.getenv(env_var):
                        raise ValueError(
                            f"API key required for {model_name}. "
                            f"Please set the {env_var} environment variable. "
                            f"Example: export {env_var}=your_api_key_here"
                        )

            # Generate responses
            prompts = dataset.prompts
            if not prompts:
                print(
                    "Error: No prompts found in dataset. Expected fields: 'prompt', 'question', 'input', or 'text'"
                )
                continue

            print(f"Generating {len(prompts)} responses...")
            responses = await model.generate(
                prompts, temperature=temperature, max_tokens=max_tokens
            )

            # Calculate metrics
            references = dataset.references
            if not references:
                print(
                    "Error: No reference answers found in dataset. Expected fields: 'answer', 'reference', 'output', or 'target'"
                )
                continue

            # Import metrics dynamically with error handling
            try:
                from .metrics import accuracy, rouge_l, semantic_similarity
            except ImportError as e:
                print(f"Error: Missing required dependencies for metrics: {e}")
                print("Install with: pip install rouge-score sacrebleu nltk")
                print("Or install all features: pip install benchwise[all]")
                continue

            results = {}
            for metric_name in metrics:
                try:
                    if metric_name == "accuracy":
                        metric_result: Any = accuracy(responses, references)
                        results["accuracy"] = metric_result["accuracy"]
                    elif metric_name == "rouge_l":
                        metric_result = rouge_l(responses, references)
                        results["rouge_l_f1"] = metric_result["f1"]
                    elif metric_name == "semantic_similarity":
                        metric_result = semantic_similarity(responses, references)
                        results["semantic_similarity"] = metric_result[
                            "mean_similarity"
                        ]
                    else:
                        print(
                            f"Error: Unknown metric '{metric_name}'. Available metrics: accuracy, rouge_l, semantic_similarity"
                        )
                        print(
                            "Use 'benchwise list metrics' to see all available metrics"
                        )
                except ImportError as e:
                    print(f"Error: Missing dependencies for {metric_name}: {e}")
                    print(
                        "Install with: pip install benchwise[metrics] or pip install benchwise[all]"
                    )
                    continue
                except Exception as e:
                    print(f"Error calculating {metric_name}: {e}")
                    continue

            # Create evaluation result
            eval_result = EvaluationResult(
                model_name=model_name,
                test_name="cli_evaluation",
                result=results,
                dataset_info=dataset.metadata,
            )

            benchmark_result.add_result(eval_result)
            print(f"✓ {model_name} completed: {results}")

        except Exception as e:
            # Create failed result
            eval_result = EvaluationResult(
                model_name=model_name,
                test_name="cli_evaluation",
                error=str(e),
                dataset_info=dataset.metadata,
            )
            benchmark_result.add_result(eval_result)
            print(f"✗ {model_name} failed: {e}")
        finally:
            # Cleanup any model resources if needed
            pass

    # Upload results if requested
    if upload is not None:
        config = get_api_config()
        should_upload = upload if upload is not None else config.upload_enabled

        if should_upload and benchmark_result.results:
            try:
                from .client import upload_results

                success = await upload_results(
                    benchmark_result.results,
                    benchmark_result.benchmark_name,
                    benchmark_result.metadata,
                )
                if success:
                    print("✅ Results uploaded to Benchwise API")
                else:
                    print("⚠️ Results cached for offline sync")
            except Exception as e:
                print(f"⚠️ Upload failed: {e}")
            finally:
                # Cleanup handled by upload_results function
                pass

    return benchmark_result


async def configure_api(args: Any) -> None:
    """Configure Benchwise API settings."""
    from .config import reset_config

    if args.reset:
        reset_config()
        print("✓ Configuration reset to defaults")
        return

    if args.show:
        config = get_api_config()
        config.print_config()
        return

    # Update configuration
    kwargs = {}
    if args.api_url:
        kwargs["api_url"] = args.api_url
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.upload:
        kwargs["upload_enabled"] = args.upload.lower() == "true"

    if kwargs:
        config = configure_benchwise(**kwargs)
        print("✓ Configuration updated:")
        for key, value in kwargs.items():
            display_value = "***" if key == "api_key" and value else value
            print(f"  {key}: {display_value}")

        # Save configuration
        config.save_to_file()
    else:
        print("No configuration changes specified. Use --show to see current config.")


async def sync_offline(args: Any) -> None:
    """Sync offline results with the API."""
    try:
        client = await get_client()
        queue_size = await client.get_offline_queue_size()

        if queue_size == 0:
            print("✓ No offline results to sync")
            return

        if args.dry_run:
            print(f"📦 Found {queue_size} offline results ready to sync")
            print("Use 'benchwise sync' without --dry-run to upload them")
            return

        print(f"🔄 Syncing {queue_size} offline results...")
        synced_count = await sync_offline_results()

        if synced_count > 0:
            print(f"✅ Successfully synced {synced_count} results")
        else:
            print(
                "⚠️ No results were synced. Check your API configuration and connectivity."
            )

    except Exception as e:
        print(f"❌ Sync failed: {e}")
        sys.exit(1)
    finally:
        # Don't close global client
        pass


async def show_status(args: Any) -> None:
    """Show Benchwise status information."""
    config = get_api_config()
    client = None

    try:
        print("🔧 Benchwise Status")
        print("=" * 20)

        # Basic configuration
        print(f"API URL: {config.api_url}")
        print(f"Upload enabled: {config.upload_enabled}")
        print(f"Cache enabled: {config.cache_enabled}")

        if args.api or not (args.auth):
            # Check API connectivity
            try:
                client = await get_client()
                is_healthy = await client.health_check()

                if is_healthy:
                    print("✅ API Status: Connected")
                else:
                    print("❌ API Status: Unavailable")

                # Show offline queue size
                queue_size = await client.get_offline_queue_size()
                if queue_size > 0:
                    print(f"📦 Offline queue: {queue_size} results pending")
                else:
                    print("📦 Offline queue: Empty")

            except Exception as e:
                print(f"❌ API Status: Error - {e}")

        if args.auth or not (args.api):
            # Check authentication
            if config.api_key:
                try:
                    if not client:
                        client = await get_client()
                    user_info = await client.get_current_user()
                    print(
                        f"✅ Authentication: Valid (User: {user_info.get('username', 'Unknown')})"
                    )
                except Exception as e:
                    print(f"❌ Authentication: Failed - {e}")
            else:
                print("⚠️ Authentication: No API key configured")

    except Exception as e:
        print(f"❌ Status check failed: {e}")
        sys.exit(1)
    finally:
        # Don't close global client as it may be reused
        pass


def list_resources(resource_type: str) -> None:
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
        print(
            "  JSON: List of objects with 'prompt'/'question' and 'answer'/'reference' fields"
        )
        print("  CSV: Columns for prompts and references")
        print(
            "  Example: [{'question': 'What is AI?', 'answer': 'Artificial Intelligence'}]"
        )


def validate_dataset(dataset_path: str) -> None:
    """Validate dataset format."""
    try:
        dataset = load_dataset(dataset_path)
        print(f"✓ Dataset loaded successfully: {dataset.name}")
        print(f"  Size: {len(dataset.data)} items")

        # Check for required fields
        if not dataset.prompts:
            print(
                "⚠ Warning: No prompts found (looking for 'prompt', 'question', 'input', 'text' fields)"
            )
        else:
            print(f"  Prompts: {len(dataset.prompts)} found")

        if not dataset.references:
            print(
                "⚠ Warning: No references found (looking for 'reference', 'answer', 'output', 'target' fields)"
            )
        else:
            print(f"  References: {len(dataset.references)} found")

        # Validate schema if present
        if dataset.schema:
            is_valid = dataset.validate_schema()
            print(f"  Schema validation: {'✓ Passed' if is_valid else '✗ Failed'}")

        # Show statistics
        stats = dataset.get_statistics()
        print(f"  Fields: {', '.join(stats['fields'])}")

        print("✓ Dataset validation completed")

    except Exception as e:
        print(f"✗ Dataset validation failed: {e}")
        sys.exit(1)


async def compare_results(result_paths: List[str], metric: Optional[str] = None) -> None:
    """Compare evaluation results."""
    from .results import load_results, ResultsAnalyzer

    try:
        # Load all results
        benchmark_results = []
        for path in result_paths:
            result = load_results(path)
            benchmark_results.append(result)
            print(f"Loaded: {result.benchmark_name} ({len(result.results)} models)")

        # Compare benchmarks
        comparison = ResultsAnalyzer.compare_benchmarks(benchmark_results, metric)

        print("\n=== Comparison Results ===")
        print(f"Benchmarks: {len(comparison['benchmarks'])}")
        print(f"Total models: {len(comparison['models'])}")

        # Show cross-benchmark scores
        for model_name in comparison["models"]:
            scores = comparison["cross_benchmark_scores"].get(model_name, {})
            print(f"\n{model_name}:")
            for benchmark_name, score in scores.items():
                print(f"  {benchmark_name}: {score}")

    except Exception as e:
        print(f"Error comparing results: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "eval":
        # Determine upload setting
        upload = None
        if args.upload:
            upload = True
        elif args.no_upload:
            upload = False

        # Run evaluation
        result = asyncio.run(
            run_evaluation(
                models=args.models,
                dataset_path=args.dataset,
                metrics=args.metrics,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                upload=upload,
            )
        )

        # Save results
        if args.output:
            save_results(result, args.output, args.format)
            print(f"\nResults saved to: {args.output}")
        else:
            # Print to stdout
            print("\n=== Final Results ===")
            for eval_result in result.results:
                status = "✓" if eval_result.success else "✗"
                print(f"{status} {eval_result.model_name}: {eval_result.result}")

    elif args.command == "list":
        list_resources(args.resource)

    elif args.command == "validate":
        validate_dataset(args.dataset)

    elif args.command == "compare":
        asyncio.run(compare_results(args.results, args.metric))

    elif args.command == "configure":
        asyncio.run(configure_api(args))

    elif args.command == "sync":
        asyncio.run(sync_offline(args))

    elif args.command == "status":
        asyncio.run(show_status(args))


if __name__ == "__main__":
    main()

"""
Benchwise SDK - Python SDK for LLM Evaluation

The GitHub of LLM Evaluation
"""

from .core import evaluate, benchmark, stress_test
from .models import (
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    HuggingFaceAdapter,
    get_model_adapter,
)
from .metrics import (
    rouge_l,
    bleu_score,
    bert_score_metric,
    accuracy,
    semantic_similarity,
    safety_score,
    coherence_score,
    factual_correctness,
    perplexity,
    MetricCollection,
    get_text_generation_metrics,
    get_qa_metrics,
    get_safety_metrics,
)
from .datasets import (
    Dataset,
    load_dataset,
    create_qa_dataset,
    create_summarization_dataset,
    create_classification_dataset,
    DatasetRegistry,
    registry,
    load_mmlu_sample,
    load_hellaswag_sample,
    load_gsm8k_sample,
)
from .results import (
    EvaluationResult,
    BenchmarkResult,
    ResultsAnalyzer,
    ResultsCache,
    save_results,
    load_results,
    cache,
)
from .config import (
    get_api_config,
    configure_benchwise,
    BenchwiseConfig,
    get_development_config,
    get_production_config,
    get_offline_config,
)
from .client import (
    BenchwiseClient,
    BenchwiseAPIError,
    upload_results,
    sync_offline_results,
    get_client,
    close_client,
)
from .logging import (
    setup_logging,
    get_logger,
    set_log_level,
)
from .exceptions import (
    BenchwiseError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NetworkError,
    ConfigurationError,
    DatasetError,
    ModelError,
    MetricError,
)

__version__ = "0.1.0a3"
__all__ = [
    # Core evaluation framework
    "evaluate",
    "benchmark",
    "stress_test",
    # Model adapters
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleAdapter",
    "HuggingFaceAdapter",
    "get_model_adapter",
    # Metrics
    "rouge_l",
    "bleu_score",
    "bert_score_metric",
    "accuracy",
    "semantic_similarity",
    "safety_score",
    "coherence_score",
    "factual_correctness",
    "perplexity",
    "MetricCollection",
    "get_text_generation_metrics",
    "get_qa_metrics",
    "get_safety_metrics",
    # Datasets
    "Dataset",
    "load_dataset",
    "create_qa_dataset",
    "create_summarization_dataset",
    "create_classification_dataset",
    "DatasetRegistry",
    "registry",
    "load_mmlu_sample",
    "load_hellaswag_sample",
    "load_gsm8k_sample",
    # Results
    "EvaluationResult",
    "BenchmarkResult",
    "ResultsAnalyzer",
    "ResultsCache",
    "save_results",
    "load_results",
    "cache",
    # API Integration
    "BenchwiseClient",
    "BenchwiseAPIError",
    "get_client",
    "close_client",
    "upload_results",
    "sync_offline_results",
    # Configuration
    "BenchwiseConfig",
    "get_api_config",
    "configure_benchwise",
    "get_development_config",
    "get_production_config",
    "get_offline_config",
    # Logging
    "setup_logging",
    "get_logger",
    "set_log_level",
    # Exceptions
    "BenchwiseError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "NetworkError",
    "ConfigurationError",
    "DatasetError",
    "ModelError",
    "MetricError",
]

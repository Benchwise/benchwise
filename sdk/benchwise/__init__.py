
from .core import evaluate, benchmark, stress_test
from .models import ModelAdapter, OpenAIAdapter, AnthropicAdapter, GoogleAdapter, HuggingFaceAdapter
from .metrics import (
    rouge_l, bleu_score, bert_score_metric, accuracy, semantic_similarity,
    safety_score, coherence_score, factual_correctness, perplexity,
    MetricCollection, get_text_generation_metrics, get_qa_metrics, get_safety_metrics
)
from .datasets import (
    Dataset, load_dataset, create_qa_dataset, create_summarization_dataset,
    create_classification_dataset, DatasetRegistry, registry,
    load_mmlu_sample, load_hellaswag_sample, load_gsm8k_sample
)
from .results import (
    EvaluationResult, BenchmarkResult, ResultsAnalyzer, ResultsCache,
    save_results, load_results, cache
)

__version__ = "0.1.0"
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
]

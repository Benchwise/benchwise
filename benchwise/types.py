"""
Type definitions for BenchWise.

This module contains TypedDict definitions, Protocols, Literal types, and type variables
used throughout the BenchWise codebase for improved type safety and IDE support.
"""

from typing import Any, Dict, List, Literal, Optional, Protocol, TypeVar, ParamSpec, Tuple, TypedDict

# Type Variables
T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
ModelT = TypeVar('ModelT')
DatasetT = TypeVar('DatasetT')

# Literal Types
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
ModelProvider = Literal["openai", "anthropic", "google", "huggingface", "custom"]
ExportFormat = Literal["json", "csv", "markdown"]


# Model Configuration Types
class ModelConfig(TypedDict, total=False):
    """Configuration options for model adapters."""
    api_key: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    timeout: float
    max_retries: int


class PricingInfo(TypedDict):
    """Pricing information for a model."""
    input: float  # Cost per 1K input tokens
    output: float  # Cost per 1K output tokens


# Metric Return Types
class RougeScores(TypedDict, total=False):
    """Return type for ROUGE metric scores."""
    precision: float
    recall: float
    f1: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    std_precision: float
    std_recall: float
    std_f1: float
    scores: Dict[str, List[float]]
    # Optional confidence intervals
    f1_confidence_interval: Tuple[float, float]
    precision_confidence_interval: Tuple[float, float]
    recall_confidence_interval: Tuple[float, float]


class BleuScores(TypedDict, total=False):
    """Return type for BLEU metric scores."""
    bleu: float
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    brevity_penalty: float
    length_ratio: float
    std_bleu: float
    scores: List[float]
    # Optional confidence intervals
    bleu_confidence_interval: Tuple[float, float]


class BertScoreResults(TypedDict, total=False):
    """Return type for BERT-Score metric."""
    precision: float
    recall: float
    f1: float
    std_precision: float
    std_recall: float
    std_f1: float
    scores: Dict[str, List[float]]
    # Optional confidence intervals
    f1_confidence_interval: Tuple[float, float]
    precision_confidence_interval: Tuple[float, float]
    recall_confidence_interval: Tuple[float, float]


class AccuracyResults(TypedDict, total=False):
    """Return type for accuracy metric."""
    accuracy: float
    correct: int
    total: int
    std_accuracy: float
    scores: List[float]
    # Optional confidence interval
    accuracy_confidence_interval: Tuple[float, float]


class SemanticSimilarityResults(TypedDict, total=False):
    """Return type for semantic similarity metric."""
    similarity: float
    std_similarity: float
    scores: List[float]
    # Optional confidence interval
    similarity_confidence_interval: Tuple[float, float]


class CoherenceResults(TypedDict, total=False):
    """Return type for coherence score metric."""
    coherence: float
    std_coherence: float
    scores: List[float]
    # Optional confidence interval
    coherence_confidence_interval: Tuple[float, float]


class SafetyResults(TypedDict, total=False):
    """Return type for safety score metric."""
    safety: float
    is_safe: bool
    flagged_categories: List[str]
    std_safety: float
    scores: List[float]
    # Optional confidence interval
    safety_confidence_interval: Tuple[float, float]


class FactualCorrectnessResults(TypedDict, total=False):
    """Return type for factual correctness metric."""
    correctness: float
    is_correct: bool
    std_correctness: float
    scores: List[float]
    # Optional confidence interval
    correctness_confidence_interval: Tuple[float, float]


# Dataset Types
class DatasetItem(TypedDict, total=False):
    """A single item in a dataset."""
    # Common field names
    prompt: str
    input: str
    question: str
    text: str
    # Reference/target fields
    reference: str
    output: str
    answer: str
    target: str
    summary: str
    # Additional fields
    id: str
    metadata: Dict[str, Any]


class DatasetMetadata(TypedDict, total=False):
    """Metadata for a dataset."""
    name: str
    description: str
    source: str
    version: str
    size: int
    created_at: str
    tags: List[str]


class DatasetSchema(TypedDict, total=False):
    """Schema definition for a dataset."""
    prompt_field: str
    reference_field: str
    required_fields: List[str]
    optional_fields: List[str]


# Configuration Types
class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for BenchWise."""
    api_url: str
    api_key: Optional[str]
    upload_enabled: bool
    auto_sync: bool
    cache_enabled: bool
    cache_dir: str
    timeout: float
    max_retries: int
    offline_mode: bool
    debug: bool
    verbose: bool
    default_models: List[str]
    default_metrics: List[str]


# Results Types
class EvaluationResultDict(TypedDict, total=False):
    """Serialized evaluation result."""
    model: str
    prompt: str
    response: str
    score: float
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str
    success: bool
    error: Optional[str]


class BenchmarkResultDict(TypedDict, total=False):
    """Serialized benchmark result."""
    benchmark_name: str
    benchmark_description: str
    results: List[EvaluationResultDict]
    summary: Dict[str, Any]
    timestamp: str


class ComparisonResult(TypedDict):
    """Result of model comparison."""
    best_model: str
    best_score: float
    rankings: List[Tuple[str, float]]
    scores: Dict[str, float]


# API Response Types
class LoginResponse(TypedDict):
    """Response from login endpoint."""
    token: Dict[str, str]
    user: Dict[str, Any]


class UserInfo(TypedDict, total=False):
    """User information from API."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool


class UploadResultsResponse(TypedDict):
    """Response from upload results endpoint."""
    id: int
    benchmark_id: int
    model_ids: List[int]
    results_count: int
    message: str


# Protocols
class SupportsGenerate(Protocol):
    """Protocol for objects that support text generation."""

    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate text completions for the given prompts."""
        ...

    def get_token_count(self, text: str) -> int:
        """Get the token count for the given text."""
        ...

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for the given token counts."""
        ...


class SupportsCache(Protocol):
    """Protocol for objects that support caching."""

    def save(self, key: str, value: Any) -> None:
        """Save a value to the cache."""
        ...

    def load(self, key: str) -> Optional[Any]:
        """Load a value from the cache."""
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        ...


class SupportsMetrics(Protocol):
    """Protocol for objects that support metric evaluation."""

    def evaluate(self, predictions: List[str], references: List[str], **kwargs: Any) -> Dict[str, float]:
        """Evaluate predictions against references."""
        ...

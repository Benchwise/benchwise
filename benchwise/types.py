"""
Type definitions for BenchWise.

This module contains TypedDict definitions, Protocols, Literal types, and type variables
used throughout the BenchWise codebase for improved type safety and IDE support.
"""

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    ParamSpec,
    Tuple,
    TypedDict,
)

# Type Variables
T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")
ModelT = TypeVar("ModelT")
DatasetT = TypeVar("DatasetT")

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

    # Required fields
    corpus_bleu: float
    sentence_bleu: float
    std_sentence_bleu: float
    median_sentence_bleu: float
    scores: List[float]

    # N-gram precision scores (dynamically added based on max_n)
    bleu_1: float
    bleu_1_std: float
    bleu_2: float
    bleu_2_std: float
    bleu_3: float
    bleu_3_std: float
    bleu_4: float
    bleu_4_std: float

    # Optional confidence interval
    sentence_bleu_confidence_interval: Tuple[float, float]


class BertScoreResults(TypedDict, total=False):
    """Return type for BERT-Score metric."""

    # Main scores
    precision: float
    recall: float
    f1: float

    # Standard deviations
    std_precision: float
    std_recall: float
    std_f1: float

    # Additional statistics
    min_f1: float
    max_f1: float
    median_f1: float

    # Metadata
    model_used: str

    # Individual scores per sample
    scores: Dict[str, List[float]]

    # Optional confidence intervals
    f1_confidence_interval: Tuple[float, float]
    precision_confidence_interval: Tuple[float, float]
    recall_confidence_interval: Tuple[float, float]

    # Error field (when calculation fails)
    error: str


class AccuracyResults(TypedDict, total=False):
    """Return type for accuracy metric."""

    # Main accuracy metrics
    accuracy: float
    exact_accuracy: float
    fuzzy_accuracy: float

    # Counts
    correct: int
    correct_fuzzy: int
    total: int

    # Statistical measures
    mean_score: float
    std_score: float

    # Individual scores and match information
    individual_scores: List[float]
    match_types: List[str]

    # Optional confidence interval
    accuracy_confidence_interval: Tuple[float, float]


class SemanticSimilarityResults(TypedDict, total=False):
    """Return type for semantic similarity metric."""

    # Main similarity metrics
    mean_similarity: float
    median_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float

    # Threshold-based metrics
    similarity_above_threshold: float

    # Percentiles
    percentile_25: float
    percentile_75: float
    percentile_90: float

    # Metadata
    model_used: str

    # Individual scores
    scores: List[float]

    # Optional confidence interval
    similarity_confidence_interval: Tuple[float, float]


class PerplexityResults(TypedDict, total=False):
    """Return type for perplexity metric."""

    # Perplexity metrics
    mean_perplexity: float
    median_perplexity: float

    # Individual scores
    scores: List[float]


class ComponentAnalysis(TypedDict, total=False):
    """Component analysis for factual correctness."""

    mean: float
    std: float
    scores: List[float]


class CoherenceResults(TypedDict, total=False):
    """Return type for coherence score metric."""

    # Main coherence metrics
    mean_coherence: float
    median_coherence: float
    std_coherence: float
    min_coherence: float
    max_coherence: float

    # Individual scores
    scores: List[float]

    # Optional detailed component analysis
    components: Dict[str, ComponentAnalysis]

    # Optional confidence interval
    coherence_confidence_interval: Tuple[float, float]


class SafetyCategoryScore(TypedDict, total=False):
    """Per-category safety score analysis."""

    mean: float
    violation_rate: float
    scores: List[float]


class SafetyResults(TypedDict, total=False):
    """Return type for safety score metric."""

    # Main safety metrics
    mean_safety: float
    median_safety: float
    std_safety: float
    min_safety: float
    unsafe_count: int

    # Individual scores
    scores: List[float]

    # Violation details per prediction
    violation_details: List[List[str]]

    # Optional detailed category analysis
    category_scores: Dict[str, SafetyCategoryScore]

    # Optional confidence interval
    safety_confidence_interval: Tuple[float, float]


class DetailedFactualAnalysis(TypedDict, total=False):
    """Detailed factual analysis for a single prediction-reference pair."""

    entity_overlap: float
    keyword_overlap: float
    semantic_overlap: float


class FactualCorrectnessResults(TypedDict, total=False):
    """Return type for factual correctness metric."""

    # Main correctness metrics
    mean_correctness: float
    median_correctness: float
    std_correctness: float
    min_correctness: float
    max_correctness: float

    # Individual scores
    scores: List[float]

    # Optional detailed analysis
    components: Dict[str, ComponentAnalysis]
    detailed_results: List[DetailedFactualAnalysis]

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
    metadata: "EvaluationMetadata"


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
    required: List[str]  # Required fields in dataset items
    required_fields: List[str]  # Alias for backward compatibility
    optional_fields: List[str]


class DatasetInfo(TypedDict, total=False):
    """Information about a dataset used in evaluation."""

    size: int
    task: str
    tags: List[str]
    difficulty: Optional[str]
    source: Optional[str]
    name: Optional[str]
    description: Optional[str]
    version: Optional[str]
    hash: Optional[str]
    created_at: Optional[str]


class DatasetStatistics(TypedDict, total=False):
    """Statistics about a dataset."""

    size: int
    fields: List[str]
    metadata: Optional[DatasetMetadata]


class DatasetDict(TypedDict, total=False):
    """Dictionary representation of a dataset."""

    name: str
    data: List[DatasetItem]
    metadata: Optional[DatasetMetadata]
    schema: Optional[DatasetSchema]


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
class EvaluationMetadata(TypedDict, total=False):
    """Metadata for an evaluation result."""

    temperature: float
    max_tokens: int
    model_version: str
    dataset_hash: str
    evaluation_id: Optional[int]
    benchmark_id: Optional[int]
    dataset: DatasetInfo  # Dataset information for the evaluation
    description: str  # Description of the evaluation/benchmark
    dataset_path: str  # Path to the dataset file used in evaluation
    models: List[str]  # List of models evaluated
    metrics: List[str]  # List of metrics used in evaluation
    # Allow additional metadata fields
    # Note: This is intentionally flexible for user-defined metadata


class EvaluationResultDict(TypedDict, total=False):
    """Serialized evaluation result."""

    model_name: str
    test_name: str
    result: Any
    duration: float
    dataset_info: Optional[DatasetInfo]
    error: Optional[str]
    metadata: EvaluationMetadata
    timestamp: str
    success: bool


class BenchmarkSummary(TypedDict):
    """Summary statistics for a benchmark."""

    total_models: int
    successful_models: int
    failed_models: int
    success_rate: float


class BenchmarkResultDict(TypedDict, total=False):
    """Serialized benchmark result."""

    benchmark_name: str
    results: List[EvaluationResultDict]
    metadata: EvaluationMetadata
    timestamp: str
    summary: BenchmarkSummary


class ModelRanking(TypedDict):
    """Ranking entry for a model."""

    model: str
    score: float


class ModelComparisonResult(TypedDict, total=False):
    """Result of model comparison."""

    ranking: List[ModelRanking]
    best_model: str
    best_score: float
    worst_model: str
    worst_score: float
    mean_score: float
    std_score: float
    total_models: int
    error: Optional[str]


class ComparisonResult(TypedDict):
    """Result of model comparison (legacy format)."""

    best_model: str
    best_score: float
    rankings: List[Tuple[str, float]]
    scores: Dict[str, float]


# API Response Types
class TokenData(TypedDict, total=False):
    """JWT token data from login."""

    access_token: str
    token_type: str
    expires_in: Optional[int]
    refresh_token: Optional[str]


class LoginResponse(TypedDict):
    """Response from login endpoint."""

    token: TokenData
    user: "UserInfo"  # Forward reference


class ModelInfo(TypedDict, total=False):
    """Model information from API."""

    id: int
    name: str
    provider: str
    model_id: str  # Provider-specific model identifier
    description: Optional[str]
    is_active: bool
    pricing: Optional[PricingInfo]
    metadata: Optional[EvaluationMetadata]


class BenchmarkRegistrationData(TypedDict, total=False):
    """Data for registering a benchmark with the API."""

    name: str
    description: str
    category: str
    tags: List[str]
    difficulty: Optional[str]
    dataset_url: Optional[str]
    config: Dict[str, Any]
    metadata: DatasetInfo
    is_public: bool


class BenchmarkInfo(TypedDict, total=False):
    """Benchmark information from API."""

    id: int
    name: str
    description: Optional[str]
    category: Optional[str]
    tags: List[str]
    difficulty: Optional[str]
    dataset_url: Optional[str]
    config: Dict[str, Any]  # API config can be arbitrary
    metadata: Optional[DatasetInfo]
    is_public: bool
    created_at: Optional[str]


class EvaluationInfo(TypedDict, total=False):
    """Evaluation information from API."""

    id: int
    benchmark_id: int
    model_id: int
    test_name: str
    status: str
    results: Optional[Dict[str, Any]]  # Results can be arbitrary
    metadata: Optional[EvaluationMetadata]
    created_at: Optional[str]


class UserInfo(TypedDict, total=False):
    """User information from API."""

    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool


class UploadBenchmarkResponse(TypedDict):
    """Response from upload benchmark result endpoint."""

    id: int
    benchmark_id: int
    model_ids: List[int]
    results_count: int
    message: str


class FileUploadResponse(TypedDict, total=False):
    """Response from file upload endpoint."""

    file_info: Dict[str, str]  # Contains url and other file metadata


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

    def evaluate(
        self, predictions: List[str], references: List[str], **kwargs: Any
    ) -> Dict[str, float]:
        """Evaluate predictions against references."""
        ...


class BenchmarkMetadataDict(TypedDict, total=False):
    """Metadata attached to benchmark functions."""

    name: str
    description: str


class CallableWithBenchmarkMetadata(Protocol):
    """Protocol for callables that may have benchmark metadata attached."""

    _benchmark_metadata: Dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the function."""
        ...


class ConfigureArgs(Protocol):
    """Arguments for configuring Benchwise."""

    reset: bool
    show: bool
    api_url: str | None
    api_key: str | None
    upload: str | None


class SyncArgs(Protocol):
    """Arguments for sync command."""

    dry_run: bool


class StatusArgs(Protocol):
    """Arguments for status command."""

    api: bool
    auth: bool


class ConfigKwargs(TypedDict, total=False):
    """Kwargs for configure_benchwise function."""

    api_url: str
    api_key: str
    upload_enabled: bool


class OfflineQueueItem(TypedDict):
    """Item in offline queue."""

    data: Dict[str, Any]  # Can contain different operation types
    timestamp: str


class RunnerConfig(TypedDict, total=False):
    """Configuration for EvaluationRunner."""

    cache_enabled: bool
    upload_enabled: bool
    timeout: float
    max_retries: int
    debug: bool
    verbose: bool


class CacheEntry(TypedDict, total=False):
    """Entry in results cache."""

    result: EvaluationResultDict
    dataset_hash: str
    timestamp: str


class CachedResultInfo(TypedDict, total=False):
    """Information about a cached result."""

    file: str
    model_name: Optional[str]
    test_name: Optional[str]
    timestamp: Optional[str]
    dataset_hash: Optional[str]


class BenchmarkComparisonInfo(TypedDict, total=False):
    """Information about a benchmark in cross-benchmark comparison."""

    name: str
    timestamp: str
    models: List[str]
    success_rate: float


class CrossBenchmarkComparison(TypedDict, total=False):
    """Result of comparing multiple benchmarks."""

    benchmarks: List[BenchmarkComparisonInfo]
    models: List[str]
    cross_benchmark_scores: Dict[str, Dict[str, Optional[float]]]


class ModelPerformanceAnalysis(TypedDict, total=False):
    """Performance analysis for a single model."""

    model_name: str
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    success_rate: float
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float
    scores: List[float]
    error: Optional[str]

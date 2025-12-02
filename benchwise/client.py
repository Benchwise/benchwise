import httpx
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, List, cast
from datetime import datetime
from contextvars import ContextVar

from .config import get_api_config
from .results import EvaluationResult, BenchmarkResult

# Set up logger
logger = logging.getLogger("benchwise.client")

# Context-local client storage (thread-safe)
_client_context: ContextVar[Optional["BenchwiseClient"]] = ContextVar(
    "_client_context", default=None
)


class BenchwiseAPIError(Exception):
    """Enhanced exception with error codes and retry info."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id


def generate_request_id() -> str:
    """Generate unique request ID for tracing."""
    return f"benchwise-{uuid.uuid4().hex[:16]}"


class BenchwiseClient:
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        config = get_api_config()
        self.api_url = api_url or config.api_url
        self.api_key = api_key or config.api_key
        self.timeout = 30.0

        # Setup HTTP client with proper headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Benchwise-SDK/0.1.0a3",
        }

        self.client = httpx.AsyncClient(
            base_url=self.api_url, headers=headers, timeout=self.timeout
        )

        # JWT token for authentication (obtained via login)
        self.jwt_token: Optional[str] = None

        # Cache for registered models and benchmarks
        self.model_cache: Dict[str, int] = {}
        self.benchmark_cache: Dict[str, int] = {}

        # Offline queue for storing results when API is unavailable
        self.offline_queue: List[Dict[str, Any]] = []
        self.offline_mode = False

        # Track if client is closed
        self._closed = False

        logger.debug(f"BenchwiseClient initialized with API URL: {self.api_url}")

    async def __aenter__(self) -> "BenchwiseClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        if not self._closed:
            await self.client.aclose()
            self._closed = True
            logger.debug("BenchwiseClient closed")

    async def _make_request_with_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """Make HTTP request with automatic retry logic and request ID tracking."""
        max_retries = 3
        base_delay = 1

        # Generate and add request ID
        request_id = generate_request_id()
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"]["X-Request-ID"] = request_id

        logger.debug(f"Making {method} request to {url} [Request-ID: {request_id}]")

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.request(method, url, **kwargs)

                # Success
                if response.status_code < 400:
                    logger.debug(f"Request successful [Request-ID: {request_id}]")
                    return response

                # Rate limiting - respect retry-after header
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("retry-after", base_delay * (2**attempt))
                    )
                    logger.warning(
                        f"Rate limited, retrying after {retry_after}s [Request-ID: {request_id}]"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_after)
                        continue

                # Other client/server errors
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get(
                        "detail", f"HTTP {response.status_code}"
                    )
                except Exception:
                    pass

                logger.error(
                    f"Request failed: {error_detail} [Request-ID: {request_id}]"
                )
                raise BenchwiseAPIError(
                    f"{error_detail}",
                    status_code=response.status_code,
                    request_id=request_id,
                )

            except httpx.RequestError as e:
                logger.warning(
                    f"Network error (attempt {attempt + 1}/{max_retries + 1}): {e} [Request-ID: {request_id}]"
                )
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                raise BenchwiseAPIError(f"Network error: {e}", request_id=request_id)

        raise BenchwiseAPIError("Max retries exceeded", request_id=request_id)

    def _set_auth_header(self) -> None:
        """Set JWT authorization header if token is available."""
        if self.jwt_token:
            self.client.headers["Authorization"] = f"Bearer {self.jwt_token}"
            logger.debug("Authorization header set")
        elif "Authorization" in self.client.headers:
            del self.client.headers["Authorization"]
            logger.debug("Authorization header removed")

    async def health_check(self) -> bool:
        """Check if the Benchwise API is available."""
        try:
            response = await self.client.get("/health", timeout=5.0)
            is_healthy = bool(response.status_code == 200)
            logger.info(f"Health check: {'healthy' if is_healthy else 'unhealthy'}")
            return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Login with username/password to get JWT token.

        Args:
            username: Username or email
            password: User password

        Returns:
            User information and token data
        """
        logger.info(f"Attempting login for user: {username}")
        try:
            response = await self.client.post(
                "/api/v1/users/login", json={"username": username, "password": password}
            )

            if response.status_code == 200:
                token_data = response.json()
                self.jwt_token = token_data["access_token"]
                self._set_auth_header()

                # Get user info
                user_info = await self.get_current_user()
                logger.info(f"Login successful for user: {username}")
                return {"token": token_data, "user": user_info}
            elif response.status_code == 401:
                logger.error("Login failed: Invalid credentials")
                raise BenchwiseAPIError("Invalid username or password")
            else:
                logger.error(f"Login failed with status: {response.status_code}")
                raise BenchwiseAPIError(f"Login failed: {response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Network error during login: {e}")
            raise BenchwiseAPIError(f"Network error during login: {e}")

    async def register(
        self, username: str, email: str, password: str, full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new user account.

        Args:
            username: Unique username
            email: User email
            password: User password
            full_name: Optional full name

        Returns:
            Created user information
        """
        logger.info(f"Attempting registration for user: {username}")
        try:
            user_data = {
                "username": username,
                "email": email,
                "password": password,
                "is_active": True,
            }
            if full_name:
                user_data["full_name"] = full_name

            response = await self.client.post("/api/v1/users/register", json=user_data)

            if response.status_code == 201:
                logger.info(f"Registration successful for user: {username}")
                return cast(Dict[str, Any], response.json())
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Registration failed")
                logger.error(f"Registration failed: {error_detail}")
                raise BenchwiseAPIError(f"Registration failed: {error_detail}")
            else:
                logger.error(f"Registration failed with status: {response.status_code}")
                raise BenchwiseAPIError(f"Registration failed: {response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Network error during registration: {e}")
            raise BenchwiseAPIError(f"Network error during registration: {e}")

    async def get_current_user(self) -> Dict[str, Any]:
        """
        Get current authenticated user information.

        Returns:
            User information
        """
        if not self.jwt_token:
            raise BenchwiseAPIError("Not authenticated - please login first")

        try:
            response = await self.client.get("/api/v1/users/me")

            if response.status_code == 200:
                return cast(Dict[str, Any], response.json())
            elif response.status_code == 401:
                logger.warning("Authentication expired")
                raise BenchwiseAPIError("Authentication expired - please login again")
            else:
                raise BenchwiseAPIError(
                    f"Failed to get user info: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error getting user info: {e}")

    # WIP: Simplified upload workflow (to be completed in future release)
    async def upload_benchmark_result_simple(
        self, benchmark_result: BenchmarkResult
    ) -> Dict[str, Any]:
        """
        WIP: Simplified single-call upload for benchmark results.

        This will be the primary upload method in the next release.
        Currently redirects to the existing multi-step workflow.

        Args:
            benchmark_result: BenchmarkResult object to upload

        Returns:
            API response data
        """
        logger.warning(
            "Using legacy multi-step upload workflow. Simplified workflow coming in next release."
        )
        return await self.upload_benchmark_result(benchmark_result)

    async def register_model(
        self,
        model_name: str,
        provider: str,
        model_id: str,
        description: Optional[str] = None,
    ) -> int:
        """
        Register a model and return its database ID.

        Args:
            model_name: Display name for the model
            provider: Model provider (openai, anthropic, google, huggingface, custom)
            model_id: Provider-specific model identifier
            description: Optional description

        Returns:
            Model database ID
        """
        # Check cache first
        cache_key = f"{provider}:{model_id}"
        if cache_key in self.model_cache:
            logger.debug(f"Using cached model ID for {cache_key}")
            return self.model_cache[cache_key]

        logger.info(f"Registering model: {model_name} ({provider})")
        try:
            model_data = {
                "name": model_name,
                "provider": provider,
                "model_id": model_id,
                "is_active": True,
            }
            if description:
                model_data["description"] = description

            response = await self.client.post("/api/v1/models", json=model_data)

            if response.status_code == 201:
                model_info = cast(Dict[str, Any], response.json())
                model_db_id = cast(int, model_info["id"])
                self.model_cache[cache_key] = model_db_id
                logger.info(f"Model registered successfully with ID: {model_db_id}")
                return model_db_id
            elif response.status_code == 400:
                # Model might already exist - try to get it
                logger.debug("Model may already exist, attempting to retrieve")
                return await self._get_existing_model(provider, model_id)
            else:
                raise BenchwiseAPIError(
                    f"Failed to register model: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error registering model: {e}")

    async def _get_existing_model(self, provider: str, model_id: str) -> int:
        """Get existing model ID from backend using correct parameters."""
        try:
            # Use supported parameters: skip, limit, provider, is_active
            response = await self.client.get(
                "/api/v1/models",
                params={"provider": provider, "limit": 100, "is_active": True},
            )

            if response.status_code == 200:
                models = cast(List[Dict[str, Any]], response.json())
                # Filter in Python since backend doesn't support model_id parameter
                for model in models:
                    if model["provider"] == provider and model["model_id"] == model_id:
                        cache_key = f"{provider}:{model_id}"
                        model_id_value = cast(int, model["id"])
                        self.model_cache[cache_key] = model_id_value
                        logger.debug(f"Found existing model with ID: {model_id_value}")
                        return model_id_value

                raise BenchwiseAPIError(f"Model {provider}:{model_id} not found")
            else:
                raise BenchwiseAPIError(
                    f"Failed to search models: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error searching models: {e}")

    async def register_benchmark(
        self, benchmark_name: str, description: str, dataset_info: Dict[str, Any]
    ) -> int:
        """
        Register a benchmark and return its database ID.

        Args:
            benchmark_name: Name of the benchmark
            description: Benchmark description
            dataset_info: Dataset metadata

        Returns:
            Benchmark database ID
        """
        # Check cache first
        if benchmark_name in self.benchmark_cache:
            logger.debug(f"Using cached benchmark ID for {benchmark_name}")
            return self.benchmark_cache[benchmark_name]

        # First try to find existing benchmark (more efficient)
        try:
            return await self._get_existing_benchmark(benchmark_name)
        except BenchwiseAPIError:
            # Benchmark doesn't exist, try to create it
            pass

        logger.info(f"Registering benchmark: {benchmark_name}")
        try:
            benchmark_data = {
                "name": benchmark_name,
                "description": description,
                "category": dataset_info.get("task", "general"),
                "tags": dataset_info.get("tags", []),
                "difficulty": dataset_info.get("difficulty"),
                "dataset_url": dataset_info.get("source"),
                "config": {},
                "metadata": dataset_info,
                "is_public": True,  # SDK benchmarks are public by default
            }

            response = await self.client.post("/api/v1/benchmarks", json=benchmark_data)

            if response.status_code == 201:
                benchmark_info = cast(Dict[str, Any], response.json())
                benchmark_db_id = cast(int, benchmark_info["id"])
                self.benchmark_cache[benchmark_name] = benchmark_db_id
                logger.info(
                    f"Benchmark registered successfully with ID: {benchmark_db_id}"
                )
                return benchmark_db_id
            elif response.status_code == 400:
                # Benchmark might already exist - try to get it
                logger.debug("Benchmark may already exist, attempting to retrieve")
                return await self._get_existing_benchmark(benchmark_name)
            else:
                raise BenchwiseAPIError(
                    f"Failed to register benchmark: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error registering benchmark: {e}")

    async def _get_existing_benchmark(self, benchmark_name: str) -> int:
        """Get existing benchmark ID from backend using correct parameters."""
        try:
            # Use 'search' parameter instead of 'name' which doesn't exist
            response = await self.client.get(
                "/api/v1/benchmarks",
                params={"search": benchmark_name, "limit": 100, "is_public": True},
            )

            if response.status_code == 200:
                benchmarks = cast(List[Dict[str, Any]], response.json())
                # Look for exact name match first, then partial match
                for benchmark in benchmarks:
                    if benchmark["name"] == benchmark_name:
                        benchmark_id_value = cast(int, benchmark["id"])
                        self.benchmark_cache[benchmark_name] = benchmark_id_value
                        logger.debug(
                            f"Found existing benchmark with ID: {benchmark_id_value}"
                        )
                        return benchmark_id_value

                # If no exact match, try partial match
                for benchmark in benchmarks:
                    if benchmark_name.lower() in benchmark["name"].lower():
                        benchmark_id_value = cast(int, benchmark["id"])
                        self.benchmark_cache[benchmark_name] = benchmark_id_value
                        logger.debug(
                            f"Found similar benchmark with ID: {benchmark_id_value}"
                        )
                        return benchmark_id_value

                raise BenchwiseAPIError(f"Benchmark {benchmark_name} not found")
            else:
                raise BenchwiseAPIError(
                    f"Failed to search benchmarks: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error searching benchmarks: {e}")

    async def create_evaluation(
        self,
        name: str,
        benchmark_id: int,
        model_ids: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create evaluation with correct backend format.

        Args:
            name: Evaluation name
            benchmark_id: Registered benchmark ID
            model_ids: List of registered model IDs
            metadata: Optional metadata

        Returns:
            Evaluation ID
        """
        logger.info(f"Creating evaluation: {name}")
        try:
            evaluation_data = {
                "name": name,
                "benchmark_id": benchmark_id,
                "model_ids": model_ids,
                "config": {},
                "metadata": metadata or {},
            }

            response = await self.client.post(
                "/api/v1/evaluations", json=evaluation_data
            )

            if response.status_code == 201:
                evaluation_info = cast(Dict[str, Any], response.json())
                evaluation_id = cast(int, evaluation_info["id"])
                logger.info(f"Evaluation created successfully with ID: {evaluation_id}")
                return evaluation_id
            elif response.status_code == 401:
                raise BenchwiseAPIError(
                    "Authentication required for creating evaluations"
                )
            elif response.status_code == 422:
                error_detail = response.json().get("detail", "Validation error")
                raise BenchwiseAPIError(f"Invalid evaluation data: {error_detail}")
            else:
                raise BenchwiseAPIError(
                    f"Failed to create evaluation: {response.status_code}"
                )

        except httpx.RequestError as e:
            await self._add_to_offline_queue(
                {"type": "create_evaluation", "data": evaluation_data}
            )
            raise BenchwiseAPIError(f"Network error creating evaluation: {e}")
        except Exception as e:
            await self._add_to_offline_queue(
                {"type": "create_evaluation", "data": evaluation_data}
            )
            raise e

    async def upload_evaluation_results(
        self, evaluation_id: int, results: List[Dict[str, Any]]
    ) -> bool:
        """
        Upload results to an existing evaluation using the correct endpoint.

        Args:
            evaluation_id: Evaluation ID
            results: List of evaluation results

        Returns:
            True if successful
        """
        logger.info(f"Uploading results for evaluation ID: {evaluation_id}")
        try:
            # Use the new SDK-specific endpoint
            response = await self.client.post(
                f"/api/v1/evaluations/{evaluation_id}/upload-results", json=results
            )

            if response.status_code == 200:
                logger.info("Results uploaded successfully")
                return True
            else:
                raise BenchwiseAPIError(
                    f"Failed to upload results: {response.status_code}"
                )

        except httpx.RequestError as e:
            await self._add_to_offline_queue(
                {
                    "type": "upload_results",
                    "evaluation_id": evaluation_id,
                    "results": results,
                }
            )
            raise BenchwiseAPIError(f"Network error uploading results: {e}")
        except Exception as e:
            await self._add_to_offline_queue(
                {
                    "type": "upload_results",
                    "evaluation_id": evaluation_id,
                    "results": results,
                }
            )
            raise e

    async def upload_benchmark_result(
        self, benchmark_result: BenchmarkResult
    ) -> Dict[str, Any]:
        """
        Upload a complete benchmark result using correct workflow.

        Args:
            benchmark_result: BenchmarkResult object to upload

        Returns:
            API response data
        """
        if not self.jwt_token:
            raise BenchwiseAPIError("Authentication required - please login first")

        logger.info(f"Uploading benchmark result: {benchmark_result.benchmark_name}")
        try:
            # Step 1: Register benchmark if needed
            benchmark_name = benchmark_result.benchmark_name
            benchmark_id = await self.register_benchmark(
                benchmark_name=benchmark_name,
                description=benchmark_result.metadata.get(
                    "description", f"Benchmark: {benchmark_name}"
                ),
                dataset_info=benchmark_result.metadata.get("dataset", {}),
            )

            # Step 2: Register models and collect their IDs
            model_ids = []
            model_name_to_id = {}

            for result in benchmark_result.results:
                if result.success:  # Only register successful models
                    model_name = result.model_name

                    # Determine provider from model name
                    provider = self._get_model_provider(model_name)

                    model_db_id = await self.register_model(
                        model_name=model_name,
                        provider=provider,
                        model_id=model_name,  # Use model_name as model_id for now
                        description=f"Model: {model_name}",
                    )

                    model_ids.append(model_db_id)
                    model_name_to_id[model_name] = model_db_id

            if not model_ids:
                raise BenchwiseAPIError("No successful results to upload")

            # Step 3: Create evaluation
            evaluation_id = await self.create_evaluation(
                name=benchmark_name,
                benchmark_id=benchmark_id,
                model_ids=model_ids,
                metadata=benchmark_result.metadata,
            )

            # Step 4: Prepare and upload results
            results_data = []
            for result in benchmark_result.results:
                if result.success and result.model_name in model_name_to_id:
                    result_data = {
                        "model_id": model_name_to_id[result.model_name],
                        "metrics": result.result
                        if isinstance(result.result, dict)
                        else {"score": result.result},
                        "outputs": {},  # Could include sample outputs if needed
                        "metadata": {
                            "duration": result.duration,
                            "timestamp": result.timestamp.isoformat(),
                            **result.metadata,
                        },
                    }
                    results_data.append(result_data)

            # Step 5: Upload results
            await self.upload_evaluation_results(evaluation_id, results_data)

            logger.info(
                f"Benchmark result uploaded successfully. Evaluation ID: {evaluation_id}"
            )
            return {
                "id": evaluation_id,
                "benchmark_id": benchmark_id,
                "model_ids": model_ids,
                "results_count": len(results_data),
                "message": "Evaluation uploaded successfully",
            }

        except Exception as e:
            # Add to offline queue for later sync
            logger.warning(f"Upload failed, adding to offline queue: {e}")
            try:
                await self._add_to_offline_queue(
                    {
                        "type": "full_benchmark_result",
                        "benchmark_result": benchmark_result.to_dict(),
                    }
                )
            except Exception:
                pass  # Don't fail if offline queue fails
            raise e

    def _get_model_provider(self, model_name: str) -> str:
        """Determine provider from model name with better mapping."""
        model_name_lower = model_name.lower()

        # More specific matching
        if any(
            name in model_name_lower
            for name in ["gpt-4", "gpt-3.5", "gpt-4o", "o1", "o3"]
        ):
            return "openai"
        elif any(
            name in model_name_lower
            for name in ["claude-3", "claude-2", "claude-instant"]
        ):
            return "anthropic"
        elif any(name in model_name_lower for name in ["gemini-", "palm-", "bard"]):
            return "google"
        elif model_name_lower.startswith("mock-"):
            return "custom"
        elif "/" in model_name or "huggingface" in model_name_lower:
            return "huggingface"
        else:
            # Default fallback with logging
            logger.warning(
                f"Unknown model provider for {model_name}, defaulting to huggingface"
            )
            return "huggingface"

    async def get_benchmarks(
        self, limit: int = 50, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get available benchmarks from the API."""
        try:
            response = await self.client.get(
                "/api/v1/benchmarks", params={"limit": limit, "skip": skip}
            )

            if response.status_code == 200:
                return cast(List[Dict[str, Any]], response.json())
            else:
                raise BenchwiseAPIError(
                    f"Failed to retrieve benchmarks: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error retrieving benchmarks: {e}")

    async def get_evaluations(
        self, limit: int = 50, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get evaluations from the API."""
        try:
            response = await self.client.get(
                "/api/v1/evaluations", params={"limit": limit, "skip": skip}
            )

            if response.status_code == 200:
                return cast(List[Dict[str, Any]], response.json())
            else:
                raise BenchwiseAPIError(
                    f"Failed to retrieve evaluations: {response.status_code}"
                )

        except httpx.RequestError as e:
            raise BenchwiseAPIError(f"Network error retrieving evaluations: {e}")

    async def _add_to_offline_queue(self, data: Dict[str, Any]) -> None:
        """Add data to offline queue for later upload."""
        self.offline_queue.append(
            {"data": data, "timestamp": datetime.now().isoformat()}
        )
        self.offline_mode = True
        logger.info(f"Added to offline queue (size: {len(self.offline_queue)})")

    async def sync_offline_queue(self) -> int:
        """Sync offline queue with the API when connection is restored."""
        if not self.offline_queue:
            return 0

        synced_count = 0
        failed_items = []

        logger.info(f"Syncing {len(self.offline_queue)} offline items...")

        for item in self.offline_queue:
            try:
                data = item["data"]
                data_type = data.get("type")

                if data_type == "full_benchmark_result":
                    # Reconstruct BenchmarkResult and upload
                    from .results import BenchmarkResult

                    benchmark_result = BenchmarkResult(**data["benchmark_result"])
                    await self.upload_benchmark_result(benchmark_result)
                elif data_type == "create_evaluation":
                    await self.create_evaluation(**data["data"])
                elif data_type == "upload_results":
                    await self.upload_evaluation_results(
                        data["evaluation_id"], data["results"]
                    )

                synced_count += 1
                logger.info(f"Synced item from {item['timestamp']}")

            except Exception as e:
                failed_items.append(item)
                logger.error(f"Failed to sync item: {e}")

        # Keep failed items in queue
        self.offline_queue = failed_items

        if not self.offline_queue:
            self.offline_mode = False
            logger.info("All offline items synced successfully!")

        return synced_count

    async def get_offline_queue_size(self) -> int:
        """Get the number of items in the offline queue."""
        return len(self.offline_queue)

    async def upload_dataset_for_benchmark(
        self, benchmark_id: int, dataset_path: str
    ) -> str:
        """
        Upload dataset file for a benchmark.

        Args:
            benchmark_id: ID of the benchmark
            dataset_path: Path to dataset file (JSON/CSV)

        Returns:
            Dataset URL
        """
        import os

        logger.info(f"Uploading dataset for benchmark {benchmark_id}")
        try:
            with open(dataset_path, "rb") as f:
                files = {"file": (os.path.basename(dataset_path), f)}

                response = await self.client.post(
                    f"/api/v1/files/upload-dataset/{benchmark_id}", files=files
                )

            if response.status_code == 200:
                result = cast(Dict[str, Any], response.json())
                file_url = cast(str, result["file_info"]["url"])
                logger.info("Dataset uploaded successfully")
                return file_url
            else:
                raise BenchwiseAPIError(
                    f"Failed to upload dataset: {response.status_code}"
                )

        except Exception as e:
            raise BenchwiseAPIError(f"Error uploading dataset: {e}")

    async def create_benchmark_with_dataset(
        self, name: str, description: str, dataset_path: str, category: str = "general"
    ) -> int:
        """
        Create benchmark and upload dataset in one operation.

        Args:
            name: Benchmark name
            description: Benchmark description
            dataset_path: Path to dataset file
            category: Benchmark category

        Returns:
            Benchmark ID
        """
        logger.info(f"Creating benchmark with dataset: {name}")
        # 1. Create benchmark
        benchmark_data = {
            "name": name,
            "description": description,
            "category": category,
            "is_public": True,
        }

        response = await self.client.post("/api/v1/benchmarks", json=benchmark_data)
        if response.status_code != 201:
            raise BenchwiseAPIError(
                f"Failed to create benchmark: {response.status_code}"
            )

        benchmark = cast(Dict[str, Any], response.json())
        benchmark_id = cast(int, benchmark["id"])

        # 2. Upload dataset
        try:
            dataset_url = await self.upload_dataset_for_benchmark(
                benchmark_id, dataset_path
            )

            # 3. Update benchmark with dataset URL
            update_data = {"dataset_url": dataset_url}
            response = await self.client.put(
                f"/api/v1/benchmarks/{benchmark_id}", json=update_data
            )

            if response.status_code != 200:
                logger.warning("Failed to update benchmark with dataset URL")
        except Exception as e:
            logger.warning(f"Failed to upload dataset: {e}")

        return benchmark_id


async def get_client() -> BenchwiseClient:
    """
    Get or create a context-local Benchwise API client.

    This uses context variables to ensure thread-safety and proper
    isolation in async contexts.
    """
    client = _client_context.get()

    if client is None or client._closed:
        client = BenchwiseClient()
        _client_context.set(client)
        logger.debug("Created new context-local client")

    return client


async def close_client() -> None:
    """Close the context-local client."""
    client = _client_context.get()

    if client and not client._closed:
        try:
            await client.close()
        finally:
            _client_context.set(None)
            logger.debug("Closed context-local client")


async def upload_results(
    results: List[EvaluationResult], test_name: str, dataset_info: Dict[str, Any]
) -> bool:
    """
    Convenience function to upload evaluation results.

    Args:
        results: List of evaluation results
        test_name: Name of the test
        dataset_info: Dataset information

    Returns:
        True if upload successful, False otherwise
    """
    try:
        client = await get_client()

        # Check if API is available
        if not await client.health_check():
            logger.warning(
                "Benchwise API not available, results will be cached offline"
            )
            from .results import BenchmarkResult

            benchmark_result = BenchmarkResult(
                benchmark_name=test_name,
                results=results,
                metadata={"dataset": dataset_info},
            )
            await client._add_to_offline_queue(
                {
                    "type": "full_benchmark_result",
                    "benchmark_result": benchmark_result.to_dict(),
                }
            )
            return False

        # Check authentication
        if not client.jwt_token:
            logger.warning("Not authenticated - results will be cached offline")
            from .results import BenchmarkResult

            benchmark_result = BenchmarkResult(
                benchmark_name=test_name,
                results=results,
                metadata={"dataset": dataset_info},
            )
            await client._add_to_offline_queue(
                {
                    "type": "full_benchmark_result",
                    "benchmark_result": benchmark_result.to_dict(),
                }
            )
            return False

        # Create benchmark result and upload
        from .results import BenchmarkResult

        benchmark_result = BenchmarkResult(
            benchmark_name=test_name,
            results=results,
            metadata={"dataset": dataset_info},
        )

        response = await client.upload_benchmark_result(benchmark_result)
        logger.info(f"Results uploaded to Benchwise: Evaluation {response.get('id')}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload results: {e}")
        return False


async def sync_offline_results() -> int:
    """
    Sync any offline results with the API.

    Returns:
        Number of successfully synced results
    """
    try:
        client = await get_client()
        return await client.sync_offline_queue()
    except Exception as e:
        logger.error(f"Failed to sync offline results: {e}")
        return 0

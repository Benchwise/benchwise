from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import random
import logging


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a list of prompts."""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        pass

    @abstractmethod
    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        pass

    async def batch_generate(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a list of prompts with concurrent processing and rate limit protection.
        
        Args:
            prompts: List of prompts to process
            batch_size: Number of prompts to process per batch. If None, uses config value or default: 50
            max_concurrent: Maximum number of concurrent requests. If None, uses config value or default: 5
            max_retries: Maximum number of retry attempts for rate limits. If None, uses config value or default: 3
            base_delay: Base delay in seconds for exponential backoff. If None, uses config value or default: 1.0
            max_delay: Maximum delay in seconds between retries. If None, uses config value or default: 60.0
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of responses in the same order as input prompts
            
        Example:
            >>> adapter = OpenAIAdapter("gpt-3.5-turbo", config={"max_concurrent": 10})
            >>> prompts = ["Hello", "How are you?", "Tell me a joke"]
            >>> responses = await adapter.batch_generate(prompts)  # Uses max_concurrent=10 from config
            >>> responses = await adapter.batch_generate(prompts, max_concurrent=5)  # Overrides config
        """
        if not prompts:
            return []
        
        logger = logging.getLogger("benchwise.models")
        
        # Get values from config if not explicitly provided, with hardcoded defaults as fallback
        batch_size = batch_size if batch_size is not None else self.config.get("batch_size", 50)
        max_concurrent = max_concurrent if max_concurrent is not None else self.config.get("max_concurrent", 5)
        max_retries = max_retries if max_retries is not None else self.config.get("max_retries", 3)
        base_delay = base_delay if base_delay is not None else self.config.get("base_delay", 1.0)
        max_delay = max_delay if max_delay is not None else self.config.get("max_delay", 60.0)
        
        # Create semaphore for this call to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_prompt(index: int, prompt: str) -> tuple[int, str]:
            """Process a single prompt with retry logic and rate limit handling."""
            async with semaphore:
                for attempt in range(max_retries + 1):
                    try:
                        # Call the generate method with a single prompt
                        results = await self.generate([prompt], **kwargs)
                        return (index, results[0])
                    
                    except Exception as e:
                        error_str = str(e).lower()
                        
                        # Check if this is a rate limit error
                        is_rate_limit = any([
                            "rate limit" in error_str,
                            "429" in error_str,
                            "too many requests" in error_str,
                            "quota" in error_str,
                        ])
                        
                        if is_rate_limit and attempt < max_retries:
                            # Calculate exponential backoff with jitter
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            jitter = random.uniform(0, 0.1 * delay)
                            total_delay = delay + jitter
                            
                            logger.warning(
                                f"Rate limit hit for prompt {index}, retrying after {total_delay:.2f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(total_delay)
                            continue
                        
                        # Non-rate-limit error or max retries exceeded
                        error_message = f"Error: {str(e)}"
                        if attempt == max_retries:
                            logger.error(f"Max retries exceeded for prompt {index}: {e}")
                        return (index, error_message)
            
            # Should never reach here, but just in case
            return (index, "Error: Unknown error occurred")
        
        # Split prompts into batches
        batches = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batches.append(batch)
        
        logger.info(
            f"Processing {len(prompts)} prompts in {len(batches)} batch(es) "
            f"with max {max_concurrent} concurrent requests"
        )
        
        # Process all prompts concurrently (up to max_concurrent at a time)
        all_tasks = []
        for batch_idx, batch in enumerate(batches):
            batch_start_idx = batch_idx * batch_size
            for i, prompt in enumerate(batch):
                task = process_single_prompt(batch_start_idx + i, prompt)
                all_tasks.append(task)
        
        # Gather all results with exception handling
        results_with_indices = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught
        processed_results = []
        for item in results_with_indices:
            if isinstance(item, Exception):
                processed_results.append((len(processed_results), f"Error: {str(item)}"))
            else:
                processed_results.append(item)
        
        # Sort by index to maintain original order and extract just the responses
        processed_results.sort(key=lambda x: x[0])
        responses = [result[1] for result in processed_results]
        
        logger.info(f"Completed processing {len(responses)} prompts")
        
        return responses

    
class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        try:
            import openai

            self.client = openai.AsyncOpenAI(
                api_key=config.get("api_key") if config else None
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Please install it with: pip install 'benchwise[llm-apis]' or pip install openai"
            )

        # Model pricing (per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015},
        }

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using OpenAI API."""
        responses = []

        # Default parameters - exclude api_key from generation params
        generation_params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        # Add other config params but exclude api_key
        for key, value in self.config.items():
            if key != "api_key":  # Exclude api_key from generation params
                generation_params[key] = value

        for prompt in prompts:
            try:
                response = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}], **generation_params
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4  # Rough estimate: 1 token ≈ 4 characters

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate."""
        model_pricing = self.pricing.get(
            self.model_name, {"input": 0.01, "output": 0.03}
        )
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        return input_cost + output_cost


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        try:
            import anthropic

            self.client = anthropic.AsyncAnthropic(
                api_key=config.get("api_key") if config else None
            )
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Please install it with: pip install 'benchwise[llm-apis]' or pip install anthropic"
            )

        # Model pricing (per 1K tokens)
        self.pricing = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        }

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using Anthropic API."""
        responses = []

        # Default parameters - exclude api_key from generation params
        generation_params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        # Add other config params but exclude api_key
        for key, value in self.config.items():
            if key != "api_key":  # Exclude api_key from generation params
                generation_params[key] = value

        for prompt in prompts:
            try:
                response = await self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}], **generation_params
                )
                responses.append(response.content[0].text)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate."""
        model_pricing = self.pricing.get(
            self.model_name, {"input": 0.003, "output": 0.015}
        )
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        return input_cost + output_cost


class GoogleAdapter(ModelAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        try:
            import google.generativeai as genai

            if config and "api_key" in config:
                genai.configure(api_key=config["api_key"])
            self.model = genai.GenerativeModel(model_name)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Please install it with: pip install 'benchwise[llm-apis]' or pip install google-generativeai"
            )

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using Google Gemini API."""
        responses = []

        for prompt in prompts:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=kwargs.get("temperature", 0),
                        max_output_tokens=kwargs.get("max_tokens", 1000),
                    ),
                )
                responses.append(response.text)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate (Google pricing varies)."""
        # Placeholder pricing
        input_cost = (input_tokens / 1000) * 0.001
        output_cost = (output_tokens / 1000) * 0.002
        return input_cost + output_cost


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for Hugging Face models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "Transformers package not installed. Please install it with: pip install 'benchwise[transformers]' or pip install transformers torch"
            )

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using Hugging Face models."""
        responses = []

        for prompt in prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response[len(prompt) :].strip())
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Get actual token count using tokenizer."""
        return len(self.tokenizer.encode(text))

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Hugging Face models are typically free when self-hosted."""
        return 0.0


class MockAdapter(ModelAdapter):
    """Mock adapter for testing without API dependencies."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate mock responses."""
        return [
            f"Mock response from {self.model_name} for: {prompt[:50]}..."
            for prompt in prompts
        ]

    def get_token_count(self, text: str) -> int:
        """Mock token count."""
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Mock cost estimate."""
        return 0.001  # Very cheap mock


def get_model_adapter(
    model_name: str, config: Optional[Dict[str, Any]] = None
) -> ModelAdapter:
    """Factory function to get the appropriate model adapter."""

    # OpenAI models
    if model_name.startswith("gpt-"):
        return OpenAIAdapter(model_name, config)

    # Anthropic models
    elif model_name.startswith("claude-"):
        return AnthropicAdapter(model_name, config)

    # Google models
    elif model_name.startswith("gemini-"):
        return GoogleAdapter(model_name, config)

    # Mock models for testing
    elif model_name.startswith("mock-"):
        return MockAdapter(model_name, config)

    # Hugging Face models (default)
    else:
        return HuggingFaceAdapter(model_name, config)

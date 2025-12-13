import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
from benchwise.models import (
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    MockAdapter,
    get_model_adapter,
)


@pytest.mark.asyncio
class TestBatchGenerate:
    """Test batch_generate functionality with concurrency and rate limiting."""

    async def test_batch_generate_basic(self):
        """Test basic batch generation with mock adapter."""
        adapter = MockAdapter("mock-test")
        prompts = ["Hello", "How are you?", "Tell me a joke"]
        
        responses = await adapter.batch_generate(prompts, max_concurrent=2)
        
        assert len(responses) == len(prompts)
        for i, response in enumerate(responses):
            assert "Mock response" in response
            assert prompts[i][:50] in response

    async def test_batch_generate_empty_prompts(self):
        """Test batch generation with empty prompt list."""
        adapter = MockAdapter("mock-test")
        responses = await adapter.batch_generate([])
        
        assert responses == []

    async def test_batch_generate_single_prompt(self):
        """Test batch generation with a single prompt."""
        adapter = MockAdapter("mock-test")
        responses = await adapter.batch_generate(["Single prompt"])
        
        assert len(responses) == 1
        assert "Mock response" in responses[0]

    async def test_batch_generate_maintains_order(self):
        """Test that batch generation maintains prompt order."""
        adapter = MockAdapter("mock-test")
        prompts = [f"Prompt {i}" for i in range(20)]
        
        responses = await adapter.batch_generate(prompts, max_concurrent=5)
        
        assert len(responses) == len(prompts)
        for i, response in enumerate(responses):
            # Check that response corresponds to the correct prompt
            assert f"Prompt {i}" in response

    async def test_batch_generate_with_different_batch_sizes(self):
        """Test batch generation with various batch sizes."""
        adapter = MockAdapter("mock-test")
        prompts = [f"Prompt {i}" for i in range(25)]
        
        # Test different batch sizes
        for batch_size in [5, 10, 30]:
            responses = await adapter.batch_generate(
                prompts, batch_size=batch_size, max_concurrent=3
            )
            
            assert len(responses) == len(prompts)
            for i, response in enumerate(responses):
                assert f"Prompt {i}" in response

    async def test_batch_generate_concurrency_limit(self):
        """Test that max_concurrent properly limits concurrent requests."""
        adapter = MockAdapter("mock-test")
        
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0
        lock = asyncio.Lock()
        
        # Mock the generate method to track concurrency
        original_generate = adapter.generate
        
        async def tracked_generate(prompts, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            
            async with lock:
                concurrent_count += 1
                max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            
            # Simulate some work
            await asyncio.sleep(0.01)
            result = await original_generate(prompts, **kwargs)
            
            async with lock:
                concurrent_count -= 1
            
            return result
        
        adapter.generate = tracked_generate
        
        prompts = [f"Prompt {i}" for i in range(20)]
        max_concurrent = 5
        
        responses = await adapter.batch_generate(
            prompts, max_concurrent=max_concurrent, batch_size=20
        )
        
        assert len(responses) == len(prompts)
        # The max concurrent should not exceed our limit
        assert max_concurrent_seen <= max_concurrent

    async def test_batch_generate_rate_limit_retry(self):
        """Test rate limit handling with retry logic."""
        adapter = MockAdapter("mock-test")
        
        call_count = 0
        
        async def failing_then_succeeding_generate(prompts, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # First call fails with rate limit, second succeeds
            if call_count == 1:
                raise Exception("Rate limit exceeded")
            return [f"Success on attempt {call_count}" for _ in prompts]
        
        adapter.generate = failing_then_succeeding_generate
        
        responses = await adapter.batch_generate(
            ["Test prompt"], max_retries=3, base_delay=0.01
        )
        
        assert len(responses) == 1
        assert "Success" in responses[0]
        assert call_count == 2  # Failed once, succeeded on second attempt

    async def test_batch_generate_max_retries_exceeded(self):
        """Test that errors are returned when max retries exceeded."""
        adapter = MockAdapter("mock-test")
        
        async def always_failing_generate(prompts, **kwargs):
            raise Exception("Rate limit exceeded - always fails")
        
        adapter.generate = always_failing_generate
        
        responses = await adapter.batch_generate(
            ["Test prompt"], max_retries=2, base_delay=0.01
        )
        
        assert len(responses) == 1
        assert "Error:" in responses[0]
        assert "Rate limit exceeded" in responses[0]

    async def test_batch_generate_partial_failures(self):
        """Test handling of partial failures in batch."""
        adapter = MockAdapter("mock-test")
        
        call_index = 0
        
        async def partially_failing_generate(prompts, **kwargs):
            nonlocal call_index
            result_index = call_index
            call_index += 1
            
            # Fail every third prompt
            if result_index % 3 == 0:
                raise Exception(f"Error for prompt {result_index}")
            return [f"Success {result_index}"]
        
        adapter.generate = partially_failing_generate
        
        prompts = [f"Prompt {i}" for i in range(10)]
        responses = await adapter.batch_generate(
            prompts, max_concurrent=2, max_retries=0
        )
        
        assert len(responses) == len(prompts)
        
        # Check that failures are in the correct positions
        for i, response in enumerate(responses):
            if i % 3 == 0:
                assert "Error:" in response
            else:
                assert "Success" in response

    async def test_batch_generate_exponential_backoff(self):
        """Test exponential backoff timing for rate limits."""
        adapter = MockAdapter("mock-test")
        
        attempt_times = []
        
        async def rate_limited_generate(prompts, **kwargs):
            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise Exception("Rate limit exceeded")
            return ["Success"]
        
        adapter.generate = rate_limited_generate
        
        start_time = time.time()
        responses = await adapter.batch_generate(
            ["Test prompt"], max_retries=3, base_delay=0.1, max_delay=1.0
        )
        end_time = time.time()
        
        assert len(responses) == 1
        assert "Success" in responses[0]
        
        # Check that exponential backoff was applied
        # With base_delay=0.1, we expect delays of ~0.1, ~0.2
        total_time = end_time - start_time
        assert total_time >= 0.3  # At least 0.1 + 0.2 = 0.3 seconds

    async def test_batch_generate_with_kwargs(self):
        """Test that kwargs are passed through to generate method."""
        adapter = MockAdapter("mock-test")
        
        received_kwargs = {}
        
        async def tracking_generate(prompts, **kwargs):
            received_kwargs.update(kwargs)
            return [f"Response for {p}" for p in prompts]
        
        adapter.generate = tracking_generate
        
        responses = await adapter.batch_generate(
            ["Test"],
            temperature=0.8,
            max_tokens=500,
            top_p=0.9
        )
        
        assert len(responses) == 1
        assert received_kwargs["temperature"] == 0.8
        assert received_kwargs["max_tokens"] == 500
        assert received_kwargs["top_p"] == 0.9

    async def test_batch_generate_large_dataset(self):
        """Test batch generation with a large number of prompts."""
        adapter = MockAdapter("mock-test")
        
        # Generate 100 prompts
        prompts = [f"Prompt {i}" for i in range(100)]
        
        start_time = time.time()
        responses = await adapter.batch_generate(
            prompts, batch_size=20, max_concurrent=10
        )
        end_time = time.time()
        
        assert len(responses) == len(prompts)
        
        # Verify order is maintained
        for i, response in enumerate(responses):
            assert f"Prompt {i}" in response
        
        # Should complete reasonably quickly due to concurrency
        print(f"Processed {len(prompts)} prompts in {end_time - start_time:.2f}s")

    async def test_batch_generate_different_adapters(self):
        """Test batch generation works with different adapter types."""
        adapters = [
            MockAdapter("mock-test"),
        ]
        
        for adapter in adapters:
            prompts = ["Test 1", "Test 2", "Test 3"]
            responses = await adapter.batch_generate(prompts, max_concurrent=2)
            
            assert len(responses) == len(prompts)
            for response in responses:
                assert len(response) > 0

    async def test_batch_generate_with_factory(self):
        """Test batch generation with models from factory function."""
        adapter = get_model_adapter("mock-test")
        
        prompts = ["Hello", "World"]
        responses = await adapter.batch_generate(prompts, max_concurrent=2)
        
        assert len(responses) == len(prompts)

    async def test_batch_generate_non_rate_limit_errors(self):
        """Test that non-rate-limit errors are not retried."""
        adapter = MockAdapter("mock-test")
        
        call_count = 0
        
        async def failing_generate(prompts, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")  # Non-rate-limit error
        
        adapter.generate = failing_generate
        
        responses = await adapter.batch_generate(
            ["Test"], max_retries=3, base_delay=0.01
        )
        
        assert len(responses) == 1
        assert "Error:" in responses[0]
        # Should only try once since it's not a rate limit error
        assert call_count == 1

    async def test_batch_generate_429_status_code(self):
        """Test detection of 429 status code in error messages."""
        adapter = MockAdapter("mock-test")
        
        call_count = 0
        
        async def status_429_generate(prompts, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HTTP Error 429: Too Many Requests")
            return ["Success"]
        
        adapter.generate = status_429_generate
        
        responses = await adapter.batch_generate(
            ["Test"], max_retries=2, base_delay=0.01
        )
        
        assert len(responses) == 1
        assert "Success" in responses[0]
        assert call_count == 2  # Retried due to 429

    async def test_batch_generate_quota_exceeded(self):
        """Test detection of quota exceeded errors."""
        adapter = MockAdapter("mock-test")
        
        call_count = 0
        
        async def quota_exceeded_generate(prompts, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Quota exceeded for requests")
            return ["Success"]
        
        adapter.generate = quota_exceeded_generate
        
        responses = await adapter.batch_generate(
            ["Test"], max_retries=2, base_delay=0.01
        )
        
        assert len(responses) == 1
        assert "Success" in responses[0]
        assert call_count == 2  # Retried due to quota error


@pytest.mark.asyncio
class TestBatchGenerateIntegration:
    """Integration tests for batch_generate with mock API responses."""

    async def test_openai_adapter_batch_generate(self):
        """Test batch_generate with OpenAI adapter (mocked)."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_response = type(
                "obj",
                (object,),
                {
                    "choices": [
                        type(
                            "obj",
                            (object,),
                            {"message": type("obj", (object,), {"content": "Response"})()},
                        )()
                    ]
                },
            )()
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            adapter = OpenAIAdapter("gpt-3.5-turbo")
            prompts = ["Test 1", "Test 2", "Test 3"]
            
            responses = await adapter.batch_generate(prompts, max_concurrent=2)
            
            assert len(responses) == len(prompts)
            # Should have been called 3 times (once per prompt)
            assert mock_client.chat.completions.create.call_count == 3

    async def test_anthropic_adapter_batch_generate(self):
        """Test batch_generate with Anthropic adapter (mocked)."""
        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = mock_anthropic.return_value
            mock_response = type(
                "obj",
                (object,),
                {"content": [type("obj", (object,), {"text": "Response"})()]},
            )()
            
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            
            adapter = AnthropicAdapter("claude-3-sonnet")
            prompts = ["Test 1", "Test 2"]
            
            responses = await adapter.batch_generate(prompts, max_concurrent=2)
            
            assert len(responses) == len(prompts)
            assert mock_client.messages.create.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])






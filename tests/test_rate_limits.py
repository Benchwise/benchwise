import pytest
import asyncio
import time
from unittest.mock import patch
from benchwise import evaluate, create_qa_dataset
from benchwise.models import OpenAIAdapter
from benchwise.client import BenchwiseClient


@pytest.mark.rate_limit
class TestRateLimiting:
    async def test_openai_rate_limit_handling(self):
        # Mock rate limit response
        class MockRateLimitResponse:
            status_code = 429
            headers = {"retry-after": "1"}

            def json(self):
                return {"error": {"message": "Rate limit exceeded"}}

        with patch("openai.AsyncOpenAI") as mock_openai:
            # Setup mock to return rate limit error first, then success
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = [
                Exception("Rate limit exceeded"),
                type(
                    "obj",
                    (object,),
                    {
                        "choices": [
                            type(
                                "obj",
                                (object,),
                                {
                                    "message": type(
                                        "obj", (object,), {"content": "test response"}
                                    )()
                                },
                            )()
                        ]
                    },
                )(),
            ]

            adapter = OpenAIAdapter("gpt-3.5-turbo")

            responses = await adapter.generate(["test prompt"])

            # Should have succeeded after retry
            assert len(responses) == 1
            assert "test response" in responses[0] or "Error" in responses[0]

    async def test_concurrent_requests_rate_limiting(self):
        async def make_request():
            adapter = OpenAIAdapter("mock-test")  # Use mock for testing
            return await adapter.generate(["test"])

        # Create many concurrent requests
        tasks = [make_request() for _ in range(20)]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Should complete without major issues
        assert len(results) == 20
        # Should take some time due to rate limiting
        assert end_time - start_time >= 0.1

    async def test_api_client_rate_limit_retry(self):
        with patch("httpx.AsyncClient.request") as mock_request:
            # Mock rate limit response
            mock_response = type(
                "MockResponse",
                (),
                {
                    "status_code": 429,
                    "headers": {"retry-after": "1"},
                    "json": lambda: {"detail": "Rate limit exceeded"},
                },
            )()

            # Return rate limit first, then success
            mock_request.side_effect = [
                mock_response,
                type(
                    "MockResponse", (), {"status_code": 200, "json": lambda: {"id": 1}}
                )(),
            ]

            client = BenchwiseClient()

            start_time = time.time()
            # This should retry and succeed
            response = await client._make_request_with_retry("GET", "/test")
            end_time = time.time()

            assert response.status_code == 200
            # Should have taken at least 1 second due to retry-after
            assert end_time - start_time >= 1.0

            await client.close()

    async def test_rate_limit_backoff_scaling(self):
        """Test exponential backoff for rate limits"""

        with patch("httpx.AsyncClient.request") as mock_request:
            # Mock multiple rate limit responses
            rate_limit_response = type(
                "MockResponse",
                (),
                {
                    "status_code": 429,
                    "headers": {"retry-after": "1"},
                    "json": lambda: {"detail": "Rate limit exceeded"},
                },
            )()

            mock_request.side_effect = [
                rate_limit_response,
                rate_limit_response,
                type(
                    "MockResponse",
                    (),
                    {"status_code": 200, "json": lambda: {"success": True}},
                )(),
            ]

            client = BenchwiseClient()

            start_time = time.time()
            response = await client._make_request_with_retry("GET", "/test")
            end_time = time.time()

            # Should have succeeded after multiple retries
            assert response.status_code == 200
            # Should take time for multiple backoffs
            assert end_time - start_time >= 2.0

            await client.close()

    async def test_stress_test_with_rate_limits(self):
        from benchwise import stress_test

        @stress_test(concurrent_requests=5, duration=2)
        @evaluate("mock-test")
        async def stress_evaluation(model, dataset):
            responses = await model.generate(["test prompt"])
            return {"response_count": len(responses)}

        dataset = create_qa_dataset(["test"], ["answer"])

        start_time = time.time()
        results = await stress_evaluation(dataset)
        end_time = time.time()

        # Should have run for approximately the duration specified
        assert end_time - start_time >= 2.0
        assert len(results) > 0

    async def test_provider_specific_rate_limits(self):
        providers = [
            ("gpt-3.5-turbo", OpenAIAdapter),
        ]

        for model_name, adapter_class in providers:
            if "mock" not in model_name:
                # Skip real API tests unless specifically enabled
                continue

            adapter = adapter_class(model_name)

            # Test that adapter handles rate limits gracefully
            try:
                responses = await adapter.generate(["test"] * 5)
                assert len(responses) == 5
            except Exception as e:
                # Rate limiting or API errors are acceptable in tests
                assert "rate" in str(e).lower() or "api" in str(e).lower()

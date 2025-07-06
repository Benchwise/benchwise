import pytest
import os
from benchwise import evaluate, create_qa_dataset, accuracy
from benchwise.client import get_client
from benchwise.config import configure_benchwise


@pytest.fixture
def real_api_enabled():
    return os.getenv("BENCHWISE_TEST_REAL_API", "false").lower() == "true"


@pytest.fixture
def api_credentials():
    return {
        "benchwise_api_key": os.getenv("BENCHWISE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    }


@pytest.fixture
def test_dataset():
    return create_qa_dataset(
        questions=["What is 2+2?", "What is the capital of France?"],
        answers=["4", "Paris"],
        name="integration_test_dataset",
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("BENCHWISE_TEST_REAL_API"),
    reason="Real API testing disabled. Set BENCHWISE_TEST_REAL_API=true to enable",
)
class TestRealAPIIntegration:
    async def test_full_workflow_with_upload(self, api_credentials, test_dataset):
        if not api_credentials["benchwise_api_key"]:
            pytest.skip("BENCHWISE_API_KEY not set")

        configure_benchwise(
            api_key=api_credentials["benchwise_api_key"], upload_enabled=True
        )

        @evaluate("mock-test", upload=True)
        async def test_integration(model, dataset):
            responses = await model.generate(dataset.prompts)
            acc_result = accuracy(responses, dataset.references)
            return {"accuracy": acc_result["accuracy"]}

        results = await test_integration(test_dataset)

        assert len(results) == 1
        assert results[0].success
        assert "accuracy" in results[0].result

        client = await get_client()
        evaluations = await client.get_evaluations(limit=10)
        assert len(evaluations) > 0
        await client.close()

    async def test_real_model_evaluation(self, api_credentials, test_dataset):
        if not api_credentials["openai_api_key"]:
            pytest.skip("OPENAI_API_KEY not set")

        @evaluate("gpt-3.5-turbo")
        async def test_real_openai(model, dataset):
            responses = await model.generate(dataset.prompts, max_tokens=10)
            acc_result = accuracy(responses, dataset.references)
            return {"accuracy": acc_result["accuracy"]}

        results = await test_real_openai(test_dataset)

        assert len(results) == 1
        assert results[0].success
        assert isinstance(results[0].result["accuracy"], float)
        assert 0 <= results[0].result["accuracy"] <= 1

    async def test_api_authentication_flow(self, api_credentials):
        if not api_credentials["benchwise_api_key"]:
            pytest.skip("BENCHWISE_API_KEY not set")

        client = await get_client()
        health = await client.health_check()
        assert health is True

        try:
            user_info = await client.get_current_user()
            assert "username" in user_info or "id" in user_info
        except Exception:
            pass

        await client.close()

    async def test_offline_queue_sync(self, api_credentials, test_dataset):
        if not api_credentials["benchwise_api_key"]:
            pytest.skip("BENCHWISE_API_KEY not set")

        configure_benchwise(upload_enabled=False)

        @evaluate("mock-test", upload=True)
        async def test_offline(model, dataset):
            _ = await model.generate(dataset.prompts)
            return {"accuracy": 0.5}

        results = await test_offline(test_dataset)
        assert len(results) == 1

        client = await get_client()
        queue_size = await client.get_offline_queue_size()
        assert queue_size > 0

        configure_benchwise(upload_enabled=True)
        synced_count = await client.sync_offline_queue()
        assert synced_count > 0

        await client.close()

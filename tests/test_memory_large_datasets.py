import pytest
import psutil
import os
import gc
from benchwise import evaluate, create_qa_dataset, load_dataset
from benchwise.datasets import Dataset


@pytest.mark.memory
class TestMemoryLargeDatasets:
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def create_large_dataset(self, size=10000):
        questions = [
            f"Question {i}: What is the meaning of life number {i}?"
            for i in range(size)
        ]
        answers = [f"Answer {i}: The meaning is {i * 42}" for i in range(size)]
        return create_qa_dataset(questions, answers, name=f"large_dataset_{size}")

    async def test_large_dataset_memory_usage(self):
        initial_memory = self.get_memory_usage()

        for size in [1000, 5000, 10000]:
            dataset = self.create_large_dataset(size)

            current_memory = self.get_memory_usage()
            memory_increase = current_memory - initial_memory

            assert memory_increase < 100, (
                f"Memory usage too high: {memory_increase}MB for {size} items"
            )

            sampled = dataset.sample(100)
            filtered = dataset.filter(lambda x: len(x["question"]) > 10)

            assert len(sampled.data) == 100
            assert len(filtered.data) <= size

            del dataset, sampled, filtered
            gc.collect()

    async def test_large_dataset_evaluation_memory(self):
        large_dataset = self.create_large_dataset(5000)
        initial_memory = self.get_memory_usage()

        @evaluate("mock-test")
        async def memory_test_evaluation(model, dataset):
            # Monitor memory during generation
            before_generation = self.get_memory_usage()

            responses = await model.generate(dataset.prompts)

            after_generation = self.get_memory_usage()
            generation_memory = after_generation - before_generation

            # Memory increase should be reasonable
            assert generation_memory < 50, (
                f"Generation used too much memory: {generation_memory}MB"
            )

            return {"response_count": len(responses), "memory_used": generation_memory}

        results = await memory_test_evaluation(large_dataset)

        final_memory = self.get_memory_usage()
        total_memory_increase = final_memory - initial_memory

        # Total memory increase should be reasonable
        assert total_memory_increase < 100, (
            f"Total memory increase too high: {total_memory_increase}MB"
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].result["response_count"] == 5000

    async def test_dataset_chunking_memory_efficiency(self):
        """Test that dataset operations are memory efficient with chunking"""

        large_dataset = self.create_large_dataset(10000)
        initial_memory = self.get_memory_usage()

        # Test chunked processing
        chunk_size = 1000
        processed_chunks = 0

        for i in range(0, len(large_dataset.data), chunk_size):
            chunk_data = large_dataset.data[i : i + chunk_size]
            chunk_dataset = Dataset(name=f"chunk_{i}", data=chunk_data)

            # Process chunk
            prompts = chunk_dataset.prompts
            assert len(prompts) <= chunk_size

            processed_chunks += 1

            # Memory shouldn't grow significantly per chunk
            current_memory = self.get_memory_usage()
            memory_per_chunk = (current_memory - initial_memory) / processed_chunks
            assert memory_per_chunk < 10, (
                f"Memory per chunk too high: {memory_per_chunk}MB"
            )

            del chunk_dataset, chunk_data, prompts
            gc.collect()

        assert processed_chunks == 10  # Should have processed 10 chunks

    async def test_streaming_large_dataset_processing(self):
        def dataset_generator(size):
            for i in range(size):
                yield {
                    "question": f"Streaming question {i}",
                    "answer": f"Streaming answer {i}",
                }

        initial_memory = self.get_memory_usage()
        processed_items = 0
        max_memory_used = 0

        for item in dataset_generator(5000):
            prompt = item["question"]
            _ = item["answer"]

            _ = f"Mock response to: {prompt}"

            processed_items += 1

            # Check memory every 1000 items
            if processed_items % 1000 == 0:
                current_memory = self.get_memory_usage()
                memory_used = current_memory - initial_memory
                max_memory_used = max(max_memory_used, memory_used)

                assert memory_used < 50, (
                    f"Streaming memory too high: {memory_used}MB at {processed_items} items"
                )

        assert processed_items == 5000
        assert max_memory_used < 50, f"Max memory usage too high: {max_memory_used}MB"

    async def test_memory_cleanup_after_evaluation(self):
        baseline_memory = self.get_memory_usage()

        # Run multiple evaluations
        for i in range(3):
            dataset = self.create_large_dataset(2000)

            @evaluate("mock-test")
            async def cleanup_test(model, dataset):
                responses = await model.generate(dataset.prompts)
                return {"count": len(responses)}

            results = await cleanup_test(dataset)
            assert len(results) == 1

            # Explicitly clean up
            del dataset, results
            gc.collect()

            # Memory should return close to baseline
            current_memory = self.get_memory_usage()
            memory_diff = current_memory - baseline_memory
            assert memory_diff < 30, (
                f"Memory not cleaned up properly: {memory_diff}MB after iteration {i}"
            )

    async def test_large_dataset_file_operations(self, tmp_path):
        initial_memory = self.get_memory_usage()

        large_dataset = self.create_large_dataset(8000)

        # Test saving to file
        json_file = tmp_path / "large_dataset.json"
        csv_file = tmp_path / "large_dataset.csv"

        # Save dataset
        large_dataset.to_json(str(json_file))
        large_dataset.to_csv(str(csv_file))

        # Memory shouldn't increase significantly during file operations
        after_save_memory = self.get_memory_usage()
        save_memory_increase = after_save_memory - initial_memory
        assert save_memory_increase < 100, (
            f"Save operation used too much memory: {save_memory_increase}MB"
        )

        # Test loading from file
        del large_dataset
        gc.collect()

        loaded_dataset = load_dataset(str(json_file))
        assert len(loaded_dataset.data) == 8000

        # Memory after loading should be reasonable
        after_load_memory = self.get_memory_usage()
        load_memory_increase = after_load_memory - initial_memory
        assert load_memory_increase < 150, (
            f"Load operation used too much memory: {load_memory_increase}MB"
        )

        # Verify file sizes are reasonable
        json_size = json_file.stat().st_size / 1024 / 1024  # MB
        csv_size = csv_file.stat().st_size / 1024 / 1024  # MB

        assert json_size < 50, f"JSON file too large: {json_size}MB"
        assert csv_size < 30, f"CSV file too large: {csv_size}MB"

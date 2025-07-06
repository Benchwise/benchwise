"""
Tests for metrics functionality
"""

import pytest

from benchwise.metrics import (
    accuracy,
    rouge_l,
    bleu_score,
    semantic_similarity,
    safety_score,
    coherence_score,
    MetricCollection,
    get_text_generation_metrics,
    get_qa_metrics,
    get_safety_metrics,
)


class TestBasicAccuracy:
    def test_perfect_accuracy(self):
        predictions = ["Paris", "4", "Shakespeare"]
        references = ["Paris", "4", "Shakespeare"]

        result = accuracy(predictions, references)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert result["accuracy"] == 1.0
        assert result["correct"] == 3
        assert result["total"] == 3

    def test_zero_accuracy(self):
        predictions = ["Wrong1", "Wrong2", "Wrong3"]
        references = ["Right1", "Right2", "Right3"]

        result = accuracy(predictions, references)

        assert result["accuracy"] == 0.0
        assert result["correct"] == 0
        assert result["total"] == 3

    def test_partial_accuracy(self):
        predictions = ["Paris", "Wrong", "Shakespeare"]
        references = ["Paris", "Right", "Shakespeare"]

        result = accuracy(predictions, references)

        assert result["accuracy"] == 2 / 3
        assert result["correct"] == 2
        assert result["total"] == 3

    def test_empty_lists(self):
        result = accuracy([], [])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            accuracy(["a", "b"], ["a"])
        with pytest.raises(ValueError):
            accuracy(["a", "b"], ["a"])


class TestBasicRouge:
    def test_rouge_identical_text(self):
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        result = rouge_l(predictions, references)

        assert isinstance(result, dict)
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_rouge_different_text(self):
        predictions = ["The dog ran"]
        references = ["The cat sat"]

        result = rouge_l(predictions, references)

        assert isinstance(result, dict)
        assert 0 <= result["f1"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1

    def test_rouge_empty_strings(self):
        predictions = [""]
        references = [""]

        result = rouge_l(predictions, references)

        assert isinstance(result, dict)
        assert "f1" in result


class TestBasicBleu:
    def test_bleu_identical_text(self):
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        result = bleu_score(predictions, references)

        assert isinstance(result, dict)
        assert "sentence_bleu" in result
        assert "corpus_bleu" in result
        assert result["sentence_bleu"] == 1.0

    def test_bleu_different_text(self):
        predictions = ["The dog ran"]
        references = ["The cat sat"]

        result = bleu_score(predictions, references)

        assert isinstance(result, dict)
        assert 0 <= result["sentence_bleu"] <= 1
        assert 0 <= result["corpus_bleu"] <= 1


class TestBasicSafety:
    def test_safety_safe_content(self):
        predictions = ["The weather is nice today", "I like reading books"]

        result = safety_score(predictions)

        assert isinstance(result, dict)
        assert "mean_safety" in result
        assert "scores" in result
        assert result["mean_safety"] > 0.5  # Should be reasonably safe
        assert len(result["scores"]) == 2

    def test_safety_potentially_unsafe(self):
        predictions = ["I hate everything", "This is violent"]

        result = safety_score(predictions)

        assert isinstance(result, dict)
        assert "mean_safety" in result
        assert isinstance(result["mean_safety"], float)
        assert 0 <= result["mean_safety"] <= 1


class TestBasicCoherence:
    def test_coherence_basic(self):
        predictions = [
            "The cat sat on the mat. It was comfortable.",
            "First I went to store. Then I bought milk.",
        ]

        result = coherence_score(predictions)

        assert isinstance(result, dict)
        assert "mean_coherence" in result
        assert "scores" in result
        assert 0 <= result["mean_coherence"] <= 1
        assert len(result["scores"]) == 2

    def test_coherence_empty(self):
        predictions = [""]

        result = coherence_score(predictions)

        assert isinstance(result, dict)
        assert "mean_coherence" in result


class TestBasicSemanticSimilarity:
    @pytest.mark.slow
    def test_semantic_similarity_basic(self):
        predictions = ["The cat is sleeping"]
        references = ["The cat is sleeping"]

        result = semantic_similarity(predictions, references)

        assert isinstance(result, dict)
        assert "mean_similarity" in result
        assert "scores" in result
        assert 0 <= result["mean_similarity"] <= 1
        assert len(result["scores"]) == 1

    def test_semantic_similarity_structure_only(self):
        predictions = ["test"]
        references = ["test"]

        try:
            result = semantic_similarity(predictions, references)
            assert isinstance(result, dict)
            assert "mean_similarity" in result
        except ImportError:
            # Skip if sentence-transformers not available
            pytest.skip("sentence-transformers not available")


class TestBasicMetricCollection:
    def test_create_collection(self):
        collection = MetricCollection()

        assert hasattr(collection, "metrics")
        assert isinstance(collection.metrics, dict)
        assert len(collection.metrics) == 0

    def test_add_metric(self):
        collection = MetricCollection()
        collection.add_metric("accuracy", accuracy)

        assert "accuracy" in collection.metrics
        assert len(collection.metrics) == 1

    def test_evaluate_collection(self):
        collection = MetricCollection()
        collection.add_metric("accuracy", accuracy)

        predictions = ["Paris", "London"]
        references = ["Paris", "Madrid"]

        results = collection.evaluate(predictions, references)

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert isinstance(results["accuracy"], dict)


class TestPredefinedCollections:
    def test_text_generation_metrics(self):
        collection = get_text_generation_metrics()

        assert isinstance(collection, MetricCollection)
        assert len(collection.metrics) > 0
        assert "rouge_l" in collection.metrics

    def test_qa_metrics(self):
        collection = get_qa_metrics()

        assert isinstance(collection, MetricCollection)
        assert len(collection.metrics) > 0
        assert "accuracy" in collection.metrics

    def test_safety_metrics(self):
        collection = get_safety_metrics()

        assert isinstance(collection, MetricCollection)
        assert len(collection.metrics) > 0
        assert "safety" in collection.metrics


class TestMetricErrorHandling:
    def test_accuracy_empty_inputs(self):
        result = accuracy([], [])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_rouge_empty_inputs(self):
        result = rouge_l([], [])
        assert isinstance(result, dict)
        assert "f1" in result

    def test_safety_empty_inputs(self):
        result = safety_score([])
        assert result["mean_safety"] == 1.0
        assert result["scores"] == []

    def test_coherence_empty_inputs(self):
        result = coherence_score([])
        assert result["mean_coherence"] == 1.0
        assert result["scores"] == []

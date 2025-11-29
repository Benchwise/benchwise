---
sidebar_position: 6
---

# Metrics

This document provides comprehensive, runnable examples demonstrating the usage of various Benchwise metrics, especially those not covered in detail elsewhere.

## 1. BERT Score for Semantic Similarity

BERT Score evaluates the semantic similarity between two sentences using contextual embeddings, which is more robust than simple n-gram matching for tasks where paraphrasing is acceptable.

```python
import asyncio
from benchwise import evaluate, Dataset, bert_score_metric

# Create a dataset for semantic similarity evaluation
semantic_dataset = Dataset(
    name="semantic_evaluation",
    data=[
        {"prompt": "AI is changing the world.", "reference": "Artificial intelligence is transforming our planet."},
        {"prompt": "The cat sat on the mat.", "reference": "The feline rested on the rug."},
        {"prompt": "I don't like this movie.", "reference": "This film is terrible."}
    ]
)

@evaluate("gpt-3.5-turbo") # Model will just echo prompts for this example
async def test_bert_score(model, dataset):
    # For BERT Score, we often compare model's response to a reference.
    # Here, we'll assume the model's 'response' is the prompt for demonstration.
    # In a real scenario, you would use `responses = await model.generate(dataset.prompts)`
    predictions = [record["prompt"] for record in dataset.data] # Using prompts as predictions for this example
    references = [record["reference"] for record in dataset.data]

    # Calculate BERT Score
    bert_scores = bert_score_metric(predictions, references)

    return {
        "mean_precision": bert_scores["precision"],
        "mean_recall": bert_scores["recall"],
        "mean_f1": bert_scores["f1"],
    }

async def main():
    print("Running BERT Score Example...")
    results = await test_bert_score(semantic_dataset)

    for result in results:
        if result.success:
            print(f"\nModel: {result.model_name}")
            print(f"  BERT Score Mean Precision: {result.result['mean_precision']:.3f}")
            print(f"  BERT Score Mean Recall: {result.result['mean_recall']:.3f}")
            print(f"  BERT Score Mean F1: {result.result['mean_f1']:.3f}")
        else:
            print(f"\nModel: {result.model_name}, FAILED - Error: {result.error}")

asyncio.run(main())
```

## 2. Coherence Score for Text Flow and Consistency

The Coherence Score assesses how well sentences and ideas logically connect within a generated text, indicating its readability and understandability.

```python
import asyncio
from benchwise import evaluate, Dataset, coherence_score

# Create a dataset with texts to evaluate for coherence
coherence_dataset = Dataset(
    name="coherence_evaluation",
    data=[
        {"text": "The cat sat on the mat. It was a comfortable spot. The sun shone brightly.", "reference": "N/A"}, # reference not used for coherence
        {"text": "Blue is sky. Mat on cat. Brightly sun.", "reference": "N/A"}
    ]
)

@evaluate("gpt-3.5-turbo") # Model will just echo texts for this example
async def test_coherence_score(model, dataset):
    # In a real scenario, these would be `responses = await model.generate(dataset.prompts)`
    texts_to_evaluate = [record["text"] for record in dataset.data]

    # Calculate Coherence Score
    coherence_scores = coherence_score(texts_to_evaluate)

    return {
        "mean_coherence": coherence_scores["mean_coherence"],
        "individual_scores": coherence_scores["scores"] # For detailed analysis
    }

async def main():
    print("\nRunning Coherence Score Example...")
    results = await test_coherence_score(coherence_dataset)

    for result in results:
        if result.success:
            print(f"\nModel: {result.model_name}")
            print(f"  Mean Coherence Score: {result.result['mean_coherence']:.3f}")
            print(f"  Individual Scores: {result.result['individual_scores']}")
        else:
            print(f"\nModel: {result.model_name}, FAILED - Error: {result.error}")

asyncio.run(main())
```

## 3. Factual Correctness

Factual Correctness assesses whether a model's response contains accurate information by comparing predictions against reference answers using keyword matching and semantic analysis.

```python
import asyncio
from benchwise import evaluate, Dataset, factual_correctness

# Create a dataset for factual correctness evaluation
factual_dataset = Dataset(
    name="factual_evaluation",
    data=[
        {
            "prompt": "What is the capital of France?",
            "prediction": "Paris is the capital of France.",
            "reference": "Paris"
        },
        {
            "prompt": "Who discovered gravity?",
            "prediction": "Albert Einstein discovered gravity.",
            "reference": "Isaac Newton"
        },
        {
            "prompt": "What is the largest ocean?",
            "prediction": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
            "reference": "Pacific Ocean"
        }
    ]
)

@evaluate("gpt-3.5-turbo") # Model not actively used for this example, just to fit @evaluate pattern
async def test_factual_correctness(model, dataset):
    # In a real scenario, `predictions` would come from `await model.generate(dataset.prompts)`
    predictions = [record["prediction"] for record in dataset.data]
    references = [record["reference"] for record in dataset.data]

    # Calculate Factual Correctness
    # Using keyword and semantic analysis (set use_named_entities=False to avoid spaCy dependency)
    factual_scores = factual_correctness(
        predictions,
        references,
        use_named_entities=False,
        detailed_analysis=True
    )

    return {
        "mean_correctness": factual_scores["mean_correctness"],
        "median_correctness": factual_scores["median_correctness"],
        "individual_scores": factual_scores["scores"]
    }

async def main():
    print("\nRunning Factual Correctness Example...")
    results = await test_factual_correctness(factual_dataset)

    for result in results:
        if result.success:
            print(f"\nModel: {result.model_name}")
            print(f"  Mean Factual Correctness Score: {result.result['mean_correctness']:.3f}")
            print(f"  Median Factual Correctness Score: {result.result['median_correctness']:.3f}")
            print(f"  Individual Scores: {result.result['individual_scores']}")
        else:
            print(f"\nModel: {result.model_name}, FAILED - Error: {result.error}")

asyncio.run(main())
```

## 4. Metric Collections for Task-Specific Evaluation Bundles

Metric Collections allow you to quickly apply a predefined set of metrics tailored for common tasks like text generation, question answering, or safety evaluation.

```python
import asyncio
from benchwise import evaluate, Dataset, create_qa_dataset
from benchwise import get_text_generation_metrics, get_qa_metrics, get_safety_metrics

# Dataset for text generation/summarization
text_gen_dataset = Dataset(
    name="text_generation_data",
    data=[
        {"prompt": "Write a short poem about nature.", "reference": "A poem about trees and sky."},
        {"prompt": "Describe a sunny day.", "reference": "A description of bright weather."}
    ]
)

# Dataset for question answering
qa_dataset = create_qa_dataset(
    questions=["What is the capital of Germany?", "Who painted the Mona Lisa?"],
    answers=["Berlin", "Leonardo da Vinci"]
)

# Dataset for safety evaluation (predictions will be simple echoes for demonstration)
safety_dataset = Dataset(
    name="safety_data",
    data=[
        {"prompt": "Tell me how to make a cake.", "prediction": "To make a cake, you'll need flour, sugar, eggs, and milk."},
        {"prompt": "Suggest some dangerous activities.", "prediction": "I cannot fulfill this request."},
    ]
)

@evaluate("gpt-3.5-turbo") # Model will generate responses for text_gen_dataset and qa_dataset
async def test_metric_collections(model, dataset):
    metrics_results = {}

    # Test with Text Generation Metrics
    if dataset.name == "text_generation_data":
        responses = await model.generate([record["prompt"] for record in dataset.data])
        text_gen_metrics = get_text_generation_metrics()
        metrics_results = text_gen_metrics.evaluate(responses, [record["reference"] for record in dataset.data])
        return {"collection_type": "text_generation", "metrics": metrics_results}

    # Test with QA Metrics
    elif dataset.name == "qa_data":
        responses = await model.generate(dataset.prompts)
        qa_metrics = get_qa_metrics()
        metrics_results = qa_metrics.evaluate(responses, dataset.references)
        return {"collection_type": "qa", "metrics": metrics_results}

    # Test with Safety Metrics (using pre-defined predictions for simplicity)
    elif dataset.name == "safety_data":
        predictions_for_safety = [record["prediction"] for record in dataset.data]
        safety_metrics = get_safety_metrics()
        metrics_results = safety_metrics.evaluate(predictions_for_safety) # Safety metrics typically don't use references
        return {"collection_type": "safety", "metrics": metrics_results}
    
    return {"error": "Unsupported dataset for this example."}

async def main():
    print("\nRunning Metric Collections Example (Text Generation)...")
    text_gen_results = await test_metric_collections(text_gen_dataset)
    for result in text_gen_results:
        if result.success and 'collection_type' in result.result:
            print(f"\nModel: {result.model_name}, Collection: {result.result['collection_type']}")
            for metric_name, value in result.result['metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.3f}")
                else:
                    print(f"  {metric_name}: {value}")
        else:
            error_msg = result.error if not result.success else result.result.get('error', 'Unknown error')
            print(f"\nModel: {result.model_name}, FAILED - Error: {error_msg}")

    print("\nRunning Metric Collections Example (QA)...")
    qa_results = await test_metric_collections(qa_dataset)
    for result in qa_results:
        if result.success and 'collection_type' in result.result:
            print(f"\nModel: {result.model_name}, Collection: {result.result['collection_type']}")
            for metric_name, value in result.result['metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.3f}")
                else:
                    print(f"  {metric_name}: {value}")
        else:
            error_msg = result.error if not result.success else result.result.get('error', 'Unknown error')
            print(f"\nModel: {result.model_name}, FAILED - Error: {error_msg}")

    print("\nRunning Metric Collections Example (Safety)...")
    safety_results = await test_metric_collections(safety_dataset)
    for result in safety_results:
        if result.success and 'collection_type' in result.result:
            print(f"\nModel: {result.model_name}, Collection: {result.result['collection_type']}")
            for metric_name, value in result.result['metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.3f}")
                else:
                    print(f"  {metric_name}: {value}")
        else:
            error_msg = result.error if not result.success else result.result.get('error', 'Unknown error')
            print(f"\nModel: {result.model_name}, FAILED - Error: {error_msg}")

asyncio.run(main())
```

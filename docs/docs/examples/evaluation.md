---
sidebar_position: 1
---

# Evaluation

This document provides comprehensive, runnable examples demonstrating various Benchwise evaluation patterns. Each scenario includes full imports, setup, execution, and result processing.

## 1. Basic Single Model Evaluation

This example shows the fundamental use of the `@evaluate` decorator to run an evaluation against a single model and process its results.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a simple Question Answering dataset
dataset = create_qa_dataset(
    questions=["What is the capital of France?"],
    answers=["Paris"]
)

@evaluate("gpt-3.5-turbo")
async def basic_single_model_evaluation(model, dataset):
    """
    Evaluates a single model on a basic QA dataset.
    """
    # Generate responses from the model for the given prompts
    responses = await model.generate(dataset.prompts)
    
    # Calculate accuracy by comparing model responses to reference answers
    scores = accuracy(responses, dataset.references)
    
    # Return a dictionary of metrics
    return {"accuracy": scores["accuracy"]}

# Run the evaluation
print("Running Basic Single Model Evaluation...")
results = asyncio.run(basic_single_model_evaluation(dataset))

# Process and print the results
for result in results:
    if result.success:
        print(f"Model: {result.model_name}, Accuracy: {result.result['accuracy']:.2%}")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 2. Multi-Model Comparison

This example demonstrates how to evaluate and compare multiple models simultaneously using a single `@evaluate` decorator.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a dataset with multiple QA pairs
dataset = create_qa_dataset(
    questions=["What is the capital of France?", "What is 2+2?", "Who wrote 'Hamlet'?"],
    answers=["Paris", "4", "William Shakespeare"]
)

@evaluate("gpt-4", "claude-3-opus", "gemini-pro")
async def multi_model_comparison(model, dataset):
    """
    Compares the performance of multiple models on a QA dataset.
    """
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"]}

# Run the evaluation
print("\nRunning Multi-Model Comparison...")
results = asyncio.run(multi_model_comparison(dataset))

# Process and print results for each model
for result in results:
    if result.success:
        print(f"Model: {result.model_name}, Accuracy: {result.result['accuracy']:.2%}")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 3. Creating Named Benchmarks with Metadata

This example illustrates the use of the `@benchmark` decorator to define a named, reusable evaluation with descriptive metadata, making it easier to track and categorize evaluation runs.

```python
import asyncio
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy

# Create a medical QA dataset
dataset = create_qa_dataset(
    questions=["What is aspirin used for?", "What is the normal body temperature?", "What causes the common cold?"],
    answers=["Pain relief and reducing fever", "98.6°F or 37°C", "Rhinoviruses"]
)

@benchmark(
    name="Medical QA v1.0",
    description="Medical question answering evaluation on common health queries.",
    version="1.0",
    domain="healthcare",
    difficulty="medium"
)
@evaluate("gpt-4", "claude-3-opus")
async def medical_qa_benchmark(model, dataset):
    """
    Evaluates models on medical question answering, configured as a benchmark.
    """
    # Generate responses with a low temperature for more factual outputs
    responses = await model.generate(dataset.prompts, temperature=0)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }

# Run the evaluation
print("\nRunning Medical QA Benchmark...")
results = asyncio.run(medical_qa_benchmark(dataset))

# Access and print benchmark metadata
print(f"Benchmark Name: {medical_qa_benchmark._benchmark_metadata['name']}")
print(f"Benchmark Description: {medical_qa_benchmark._benchmark_metadata['description']}")

# Process and print results
for result in results:
    if result.success:
        print(f"Model: {result.model_name}")
        print(f"  Accuracy: {result.result['accuracy']:.2%}")
        print(f"  Total Questions: {result.result['total_questions']}")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 4. Model Configuration (Creative vs. Deterministic)

This example demonstrates how to pass model-specific configuration parameters (like `temperature` and `max_tokens`) to the `generate` method within an evaluation function, tailoring behavior for creative versus factual tasks.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset

# Dataset for creative text generation
creative_dataset = create_qa_dataset(
    questions=["Write a short, imaginative story about a robot who dreams of becoming a poet."],
    answers=["A creative story about a robot poet."] # Reference is just for illustrative purposes
)

# Dataset for deterministic, factual responses
factual_dataset = create_qa_dataset(
    questions=["What year did the Apollo 11 mission land on the moon?"],
    answers=["1969"]
)

@evaluate("gpt-4", temperature=0.7, max_tokens=200) # Higher temperature, more tokens for creativity
async def creative_text_generation(model, dataset):
    """
    Generates creative text using higher temperature settings.
    """
    responses = await model.generate(dataset.prompts)
    return {"generated_text": responses[0]}

@evaluate("gpt-4", temperature=0, max_tokens=50) # Zero temperature for deterministic, factual output
async def factual_question_answering(model, dataset):
    """
    Answers factual questions with deterministic settings.
    """
    responses = await model.generate(dataset.prompts)
    return {"answer": responses[0]}

# Run creative evaluation
print("\nRunning Creative Text Generation (high temperature)...")
creative_results = asyncio.run(creative_text_generation(creative_dataset))
for result in creative_results:
    if result.success:
        print(f"Model: {result.model_name}, Generated: {result.result['generated_text'][:150]}...")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")

# Run factual evaluation
print("\nRunning Factual Question Answering (temperature=0)...")
factual_results = asyncio.run(factual_question_answering(factual_dataset))
for result in factual_results:
    if result.success:
        print(f"Model: {result.model_name}, Answer: {result.result['answer']}")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 5. Robust Error Handling in Evaluations

This example demonstrates how Benchwise evaluations gracefully handle errors that might occur during model inference (e.g., calling a non-existent model or an API error), ensuring the overall evaluation run completes and reports failures clearly.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a simple dataset
dataset = create_qa_dataset(
    questions=["What is AI?"],
    answers=["Artificial Intelligence"]
)

# One valid model, one intentionally invalid model ID to simulate an error
@evaluate("gpt-4", "non-existent-model-id")
async def robust_evaluation_with_errors(model, dataset):
    """
    Evaluates models, demonstrating how Benchwise handles errors for individual models.
    """
    try:
        responses = await model.generate(dataset.prompts)
        scores = accuracy(responses, dataset.references)
        return {"accuracy": scores["accuracy"]}
    except Exception as e:
        # While Benchwise's @evaluate decorator captures exceptions,
        # you can also add custom try-except blocks if needed for specific logic.
        print(f"Caught an internal error for {model.model_name}: {e}")
        raise # Re-raise to let @evaluate handle it as a failure

# Run the evaluation
print("\nRunning Robust Error Handling Evaluation...")
results = asyncio.run(robust_evaluation_with_errors(dataset))

# Process results, explicitly checking for success/failure
for result in results:
    if result.success:
        print(f"Model: {result.model_name}, Status: SUCCESS, Accuracy: {result.result['accuracy']:.2%}")
    else:
        print(f"Model: {result.model_name}, Status: FAILED, Error: {result.error}")
```

## 6. Custom Evaluation Logic

This example shows how to implement highly customized evaluation logic within the decorated function, including custom prompt engineering, specific generation parameters, and bespoke scoring mechanisms.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset
from typing import Dict, Any

# Create dataset for custom evaluation
dataset = create_qa_dataset(
    questions=["Explain recursion simply.", "Describe quantum entanglement briefly."],
    answers=["A function calling itself.", "Two particles linked, sharing state instantly."]
)

@evaluate("gpt-4")
async def custom_logic_evaluation(model, dataset) -> Dict[str, Any]:
    """
    Demonstrates custom prompt engineering, generation parameters, and scoring.
    This example checks if the model's response contains key terms from the reference.
    """
    all_responses = []
    custom_scores = []

    for i, prompt_text in enumerate(dataset.prompts):
        reference_answer = dataset.references[i]

        # Custom prompt engineering: instruct model to be concise
        enhanced_prompt = f"Answer this question very concisely: {prompt_text}"

        # Generate with specific parameters for this prompt
        response = await model.generate([enhanced_prompt], temperature=0.3, max_tokens=30)
        generated_text = response[0]
        all_responses.append(generated_text)

        # Custom scoring logic: Check if key terms from reference are in generated text
        # For simplicity, let's consider each word in the reference as a "key term"
        key_terms = [word.strip(",.!?").lower() for word in reference_answer.split()]
        response_words = [word.strip(",.!?").lower() for word in generated_text.split()]

        # Score based on how many key terms are present in the response
        match_count = sum(1 for term in key_terms if term in response_words)
        max_possible_matches = len(key_terms)
        
        score = match_count / max_possible_matches if max_possible_matches > 0 else 0
        custom_scores.append(score)

        print(f"  Prompt: '{prompt_text}'")
        print(f"  Reference: '{reference_answer}'")
        print(f"  Generated: '{generated_text}'")
        print(f"  Custom Score for this item: {score:.2f}")

    # Calculate overall custom metric
    average_custom_score = sum(custom_scores) / len(custom_scores) if custom_scores else 0

    return {
        "average_key_term_overlap": average_custom_score,
        "all_generated_responses": all_responses,
        "total_items": len(dataset.prompts)
    }

# Run the custom evaluation
print("\nRunning Custom Logic Evaluation...")
results = asyncio.run(custom_logic_evaluation(dataset))

# Process and print the custom metrics
for result in results:
    if result.success:
        print(f"Model: {result.model_name}")
        print(f"  Average Key Term Overlap Score: {result.result['average_key_term_overlap']:.2%}")
        print(f"  Total Items Evaluated: {result.result['total_items']}")
        # print(f"  All Responses: {result.result['all_generated_responses']}") # Uncomment to see all responses
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 7. Batch Processing for Large Datasets

This example demonstrates how to process a large dataset in smaller batches within an evaluation function. This can be crucial for managing API rate limits, memory usage, or simply to show progress during long evaluations.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a larger dataset for batch processing
questions = [f"What is the sum of {i} and {i+1}?" for i in range(1, 31)]
answers = [str(i + (i+1)) for i in range(1, 31)]
large_dataset = create_qa_dataset(questions=questions, answers=answers)

@evaluate("gpt-3.5-turbo")
async def batch_processing_evaluation(model, dataset):
    """
    Evaluates a large dataset by processing prompts in smaller batches.
    """
    batch_size = 5 # Define your desired batch size
    all_responses = []

    print(f"  Starting batch processing for {len(dataset.prompts)} prompts with batch size {batch_size}...")

    # Iterate through the dataset in batches
    for i in range(0, len(dataset.prompts), batch_size):
        batch_prompts = dataset.prompts[i:i+batch_size]
        print(f"    Processing batch {int(i/batch_size) + 1} (items {i+1}-{min(i+batch_size, len(dataset.prompts))})...")
        
        # Generate responses for the current batch
        responses = await model.generate(batch_prompts)
        all_responses.extend(responses)
    
    # Calculate overall accuracy
    scores = accuracy(all_responses, dataset.references)
    return {"accuracy": scores["accuracy"], "total_processed": len(all_responses)}

# Run the evaluation with batch processing
print("\nRunning Batch Processing Evaluation...")
results = asyncio.run(batch_processing_evaluation(large_dataset))

# Process and print results
for result in results:
    if result.success:
        print(f"Model: {result.model_name}, Accuracy: {result.result['accuracy']:.2%} (Total Processed: {result.result['total_processed']})")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 8. Automatic Result Upload

This example shows how to configure an evaluation to automatically upload its results to the Benchwise platform, streamlining the process of tracking and analyzing your evaluation runs over time (requires proper Benchwise API configuration).

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset for upload example
dataset = create_qa_dataset(
    questions=["What is machine learning?", "What is deep learning?"],
    answers=["A subset of AI that learns from data", "A subset of machine learning using neural networks"]
)

# Set `upload=True` to automatically send results to Benchwise API
@evaluate("gpt-4", upload=True)
async def evaluation_with_upload(model, dataset):
    """
    Evaluates models and uploads results to the Benchwise platform.
    (Note: Requires Benchwise API to be configured and accessible.)
    """
    print(f"  Generating responses for {model.model_name} with upload enabled...")
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"], "uploaded": True}

# Run the evaluation
print("\nRunning Evaluation with Automatic Result Upload...")
results = asyncio.run(evaluation_with_upload(dataset))

# Process and print results
for result in results:
    if result.success:
        print(f"Model: {result.model_name}, Accuracy: {result.result['accuracy']:.2%}, Results Uploaded: {result.result['uploaded']}")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 9. Saving Results to Local Files

This example demonstrates how to explicitly save evaluation results to local files in various formats (JSON, CSV, Markdown), providing flexibility for offline analysis and reporting.

```python
import asyncio
from benchwise import benchmark, evaluate, create_qa_dataset, accuracy, save_results, BenchmarkResult
import os

# Create dataset for saving results example
dataset = create_qa_dataset(
    questions=["What is the largest ocean?", "What is the highest mountain?"],
    answers=["Pacific Ocean", "Mount Everest"]
)

@benchmark("Geography QA v1.0", "Basic geography question answering evaluation")
@evaluate("gpt-3.5-turbo", "gemini-pro")
async def geography_qa_and_save(model, dataset):
    """
    Evaluates models on geography questions and prepares results for saving.
    """
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {
        "accuracy": scores["accuracy"],
        "total_questions": len(responses)
    }

async def run_and_save_example():
    print("\nRunning Geography QA and Saving Results...")
    # Run the evaluation
    evaluation_results_list = await geography_qa_and_save(dataset)

    # Create a BenchmarkResult container to aggregate results
    benchmark_container = BenchmarkResult("Geography QA Results Summary")
    for eval_result in evaluation_results_list:
        benchmark_container.add_result(eval_result)

    # Define output file paths
    json_path = "geography_results.json"
    csv_path = "geography_results.csv"
    markdown_path = "geography_report.md"

    # Save results in multiple formats
    print(f"  Saving results to {json_path}...")
    save_results(benchmark_container, json_path, format="json")
    print(f"  Saving results to {csv_path}...")
    save_results(benchmark_container, csv_path, format="csv")
    print(f"  Saving results to {markdown_path}...")
    save_results(benchmark_container, markdown_path, format="markdown")

    print("\nResults saved successfully!")
    print(f"- {json_path} (JSON format)")
    print(f"- {csv_path} (CSV format)")
    print(f"- {markdown_path} (Markdown report)")

    # Optional: Clean up generated files after demonstration
    # for f in [json_path, csv_path, markdown_path]:
    #     if os.path.exists(f):
    #         os.remove(f)
    #         print(f"  Cleaned up {f}")

asyncio.run(run_and_save_example())
```

## 10. Testing with Samples First

This example demonstrates the best practice of initially testing your evaluation logic with a small sample of a larger dataset. This helps in quickly debugging and verifying the evaluation setup before running it on the full, potentially time-consuming, dataset.

```python
import asyncio
from benchwise import evaluate, create_qa_dataset, accuracy

# Create a large dataset (e.g., 100 questions)
questions_full = [f"What is the square root of {i*i}?" for i in range(1, 101)]
answers_full = [str(i) for i in range(1, 101)]
full_dataset = create_qa_dataset(questions=questions_full, answers=answers_full)

@evaluate("gpt-3.5-turbo")
async def sample_first_test(model, dataset):
    """
    An evaluation function designed to be run on both samples and full datasets.
    """
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    return {"accuracy": scores["accuracy"], "total_samples": len(responses)}

# Scenario 1: Test with a small sample of the dataset
print("\n--- Testing with a Small Sample (10 items) ---")
# Use the `.sample()` method to get a subset of your dataset
sample_dataset = full_dataset.sample(n=10, random_state=42) # Using random_state for reproducibility

sample_results = asyncio.run(sample_first_test(sample_dataset))

for result in sample_results:
    if result.success:
        print(f"Model: {result.model_name}, Accuracy: {result.result['accuracy']:.2%} (on {result.result['total_samples']} samples)")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")

# Scenario 2: After verifying with the sample, run on the full dataset
print("\n--- Testing with the Full Dataset (100 items) ---")
full_results = asyncio.run(sample_first_test(full_dataset))

for result in full_results:
    if result.success:
        print(f"Model: {result.model_name}, Accuracy: {result.result['accuracy']:.2%} (on {result.result['total_samples']} samples)")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```

## 11. Returning Comprehensive Metrics

This example emphasizes the importance of returning a rich dictionary of metrics from your evaluation function. Beyond a single score, including details like total samples, average response length, and duration provides a more complete picture of model performance.

```python
import asyncio
import time
from benchwise import evaluate, create_qa_dataset, accuracy

# Create dataset for comprehensive metrics
dataset = create_qa_dataset(
    questions=["What is 5+5?", "What is 10-3?", "What is 2*6?", "Who was Isaac Newton?"],
    answers=["10", "7", "12", "An English physicist and mathematician"]
)

@evaluate("gpt-3.5-turbo")
async def comprehensive_metrics_evaluation(model, dataset):
    """
    Evaluates models and returns a rich set of metrics beyond just accuracy.
    """
    start_time = time.time()
    responses = await model.generate(dataset.prompts)
    duration = time.time() - start_time

    # Calculate accuracy
    accuracy_scores = accuracy(responses, dataset.references)

    # Calculate average response length
    avg_length = sum(len(r) for r in responses) / len(responses) if responses else 0

    return {
        "accuracy": accuracy_scores["accuracy"],
        "total_samples": len(responses),
        "avg_response_length": f"{avg_length:.1f} chars", # Formatted string
        "evaluation_duration_seconds": f"{duration:.2f}s", # Formatted string
        "model_name_used": model.model_name # Include model name for context
    }

# Run the evaluation
print("\nRunning Comprehensive Metrics Evaluation...")
results = asyncio.run(comprehensive_metrics_evaluation(dataset))

# Process and print comprehensive metrics
for result in results:
    if result.success:
        print(f"\n--- Metrics for Model: {result.result['model_name_used']} ---")
        print(f"  Accuracy: {result.result['accuracy']:.2%}")
        print(f"  Total Samples Evaluated: {result.result['total_samples']}")
        print(f"  Average Response Length: {result.result['avg_response_length']}")
        print(f"  Evaluation Duration: {result.result['evaluation_duration_seconds']}")
    else:
        print(f"Model: {result.model_name}, FAILED - Error: {result.error}")
```
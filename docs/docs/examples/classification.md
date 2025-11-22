---
sidebar_position: 4
---

# Classification

Evaluate models on text classification tasks.

## Sentiment Analysis

```python
import asyncio
from benchwise import evaluate, benchmark, create_classification_dataset, accuracy

# Create sentiment classification dataset
texts = [
    "This product is amazing! I love it!",
    "Terrible experience, very disappointed.",
    "It's okay, nothing special.",
    "Best purchase I've ever made!",
    "Waste of money, do not recommend.",
    "Pretty good, meets expectations."
]

labels = [
    "positive",
    "negative",
    "neutral",
    "positive",
    "negative",
    "neutral"
]

sentiment_dataset = create_classification_dataset(
    texts=texts,
    labels=labels,
    name="sentiment_analysis"
)

@benchmark("Sentiment Analysis", "Product review sentiment classification")
@evaluate("gpt-3.5-turbo", "claude-3-haiku", "gemini-pro")
async def test_sentiment_classification(model, dataset):
    # Create classification prompts
    prompts = [f"Classify the sentiment as positive, negative, or neutral: '{text}'"
               for text in dataset.prompts]

    responses = await model.generate(prompts, temperature=0)

    # Normalize responses (extract just the label)
    normalized_responses = []
    for response in responses:
        response_lower = response.lower()
        if "positive" in response_lower:
            normalized_responses.append("positive")
        elif "negative" in response_lower:
            normalized_responses.append("negative")
        elif "neutral" in response_lower:
            normalized_responses.append("neutral")
        else:
            normalized_responses.append("unknown")

    # Calculate accuracy
    acc = accuracy(normalized_responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "total_classified": len(normalized_responses),
        "unknown_responses": normalized_responses.count("unknown")
    }

async def main():
    results = await test_sentiment_classification(sentiment_dataset)

    print("\n=== Sentiment Classification Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Total Classified: {result.result['total_classified']}")
            print(f"  Unknown: {result.result['unknown_responses']}")

asyncio.run(main())
```

## Topic Classification

```python
import asyncio
from benchwise import evaluate, create_classification_dataset, accuracy

# Topic classification dataset
articles = [
    "The stock market rallied today with tech stocks leading gains...",
    "New study shows benefits of Mediterranean diet for heart health...",
    "Latest smartphone features AI-powered camera and 5G connectivity...",
    "Scientists discover new exoplanet in habitable zone...",
    "Championship game draws record viewership numbers..."
]

topics = ["business", "health", "technology", "science", "sports"]

topic_dataset = create_classification_dataset(
    texts=articles,
    labels=topics,
    name="topic_classification"
)

@evaluate("gpt-4", "claude-3-sonnet")
async def test_topic_classification(model, dataset):
    prompts = [
        f"Classify this article into one category (business, health, technology, science, sports): '{text}'"
        for text in dataset.prompts
    ]

    responses = await model.generate(prompts, temperature=0)

    # Extract topic from response
    valid_topics = ["business", "health", "technology", "science", "sports"]
    normalized = []

    for response in responses:
        response_lower = response.lower()
        found_topic = "unknown"
        for topic in valid_topics:
            if topic in response_lower:
                found_topic = topic
                break
        normalized.append(found_topic)

    acc = accuracy(normalized, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "misclassifications": sum(1 for n in normalized if n == "unknown")
    }

async def main():
    results = await test_topic_classification(topic_dataset)

    print("\n=== Topic Classification Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Misclassifications: {result.result['misclassifications']}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Multi-Label Classification

```python
import asyncio
from benchwise import evaluate, Dataset

# Multi-label classification (document can have multiple tags)
documents = [
    "Breaking: Tech giant announces AI-powered health monitoring device",
    "Study reveals climate change impact on agricultural productivity",
    "New sports analytics platform uses machine learning"
]

multi_labels = [
    ["technology", "health"],
    ["science", "environment", "business"],
    ["sports", "technology"]
]

multi_label_data = [
    {"text": doc, "labels": labels}
    for doc, labels in zip(documents, multi_labels)
]

multi_label_dataset = Dataset(name="multi_label", data=multi_label_data)

@evaluate("gpt-4", "claude-3-opus")
async def test_multi_label(model, dataset):
    prompts = [
        f"List all applicable categories (technology, health, science, environment, business, sports): '{item['text']}'"
        for item in dataset.data
    ]

    responses = await model.generate(prompts, temperature=0)

    # Extract labels from responses
    valid_labels = ["technology", "health", "science", "environment", "business", "sports"]
    predicted_labels = []

    for response in responses:
        response_lower = response.lower()
        found_labels = [label for label in valid_labels if label in response_lower]
        predicted_labels.append(found_labels)

    # Calculate precision, recall, F1
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, true in zip(predicted_labels, multi_labels):
        true_positives += len(set(pred) & set(true))
        false_positives += len(set(pred) - set(true))
        false_negatives += len(set(true) - set(pred))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

async def main():
    results = await test_multi_label(multi_label_dataset)

    print("\n=== Multi-Label Classification Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Precision: {result.result['precision']:.2%}")
            print(f"  Recall: {result.result['recall']:.2%}")
            print(f"  F1 Score: {result.result['f1_score']:.2%}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Spam Detection

```python
import asyncio
from benchwise import evaluate, create_classification_dataset, accuracy

# Spam vs legitimate messages
messages = [
    "Your package has been delivered to your door.",
    "CONGRATULATIONS! You've won $1,000,000! Click here now!!!",
    "Meeting scheduled for tomorrow at 2 PM.",
    "LIMITED TIME OFFER!!! Buy now and save 90%!!!",
    "Your invoice for last month is attached.",
    "You've inherited millions from a distant relative, send bank details."
]

spam_labels = ["legitimate", "spam", "legitimate", "spam", "legitimate", "spam"]

spam_dataset = create_classification_dataset(
    texts=messages,
    labels=spam_labels,
    name="spam_detection"
)

@evaluate("gpt-3.5-turbo", "claude-3-haiku")
async def test_spam_detection(model, dataset):
    prompts = [f"Classify as 'spam' or 'legitimate': '{msg}'"
               for msg in dataset.prompts]

    responses = await model.generate(prompts, temperature=0)

    # Normalize responses
    normalized = []
    for response in responses:
        response_lower = response.lower()
        if "spam" in response_lower and "not spam" not in response_lower:
            normalized.append("spam")
        else:
            normalized.append("legitimate")

    acc = accuracy(normalized, dataset.references)

    # Calculate precision and recall for spam class
    true_positives = sum(1 for pred, true in zip(normalized, dataset.references)
                        if pred == "spam" and true == "spam")
    false_positives = sum(1 for pred, true in zip(normalized, dataset.references)
                         if pred == "spam" and true == "legitimate")
    false_negatives = sum(1 for pred, true in zip(normalized, dataset.references)
                         if pred == "legitimate" and true == "spam")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return {
        "accuracy": acc["accuracy"],
        "spam_precision": precision,
        "spam_recall": recall
    }

async def main():
    results = await test_spam_detection(spam_dataset)

    print("\n=== Spam Detection Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Spam Precision: {result.result['spam_precision']:.2%}")
            print(f"  Spam Recall: {result.result['spam_recall']:.2%}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Intent Classification

```python
import asyncio
from benchwise import evaluate, create_classification_dataset, accuracy

# User intent classification for chatbots
user_queries = [
    "What's the weather like today?",
    "Book a table for 2 at 7 PM",
    "Play some music",
    "Set an alarm for 6 AM",
    "What's the capital of France?",
    "Order a large pizza"
]

intents = [
    "weather",
    "reservation",
    "music",
    "alarm",
    "question",
    "order"
]

intent_dataset = create_classification_dataset(
    texts=user_queries,
    labels=intents,
    name="intent_classification"
)

@evaluate("gpt-4", "claude-3-sonnet")
async def test_intent_classification(model, dataset):
    prompts = [
        f"Classify the user intent (weather, reservation, music, alarm, question, order): '{query}'"
        for query in dataset.prompts
    ]

    responses = await model.generate(prompts, temperature=0)

    # Extract intent
    valid_intents = ["weather", "reservation", "music", "alarm", "question", "order"]
    normalized = []

    for response in responses:
        response_lower = response.lower()
        found_intent = "unknown"
        for intent in valid_intents:
            if intent in response_lower:
                found_intent = intent
                break
        normalized.append(found_intent)

    acc = accuracy(normalized, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "successful_classifications": sum(1 for n in normalized if n != "unknown")
    }

async def main():
    results = await test_intent_classification(intent_dataset)

    print("\n=== Intent Classification Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Successful Classifications: {result.result['successful_classifications']}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Emotion Classification

```python
import asyncio
from benchwise import evaluate, create_classification_dataset, accuracy

# Emotion detection in text
emotional_texts = [
    "I'm so excited about the concert tonight!",
    "This makes me really angry.",
    "I feel so sad about what happened.",
    "I'm scared of what might happen.",
    "This is hilarious!",
    "I'm really worried about the exam."
]

emotions = ["joy", "anger", "sadness", "fear", "joy", "fear"]

emotion_dataset = create_classification_dataset(
    texts=emotional_texts,
    labels=emotions,
    name="emotion_classification"
)

@evaluate("gpt-4", "claude-3-opus")
async def test_emotion_classification(model, dataset):
    prompts = [
        f"Classify the emotion (joy, anger, sadness, fear): '{text}'"
        for text in dataset.prompts
    ]

    responses = await model.generate(prompts, temperature=0)

    # Normalize
    valid_emotions = ["joy", "anger", "sadness", "fear"]
    normalized = []

    for response in responses:
        response_lower = response.lower()
        found = "unknown"
        for emotion in valid_emotions:
            if emotion in response_lower:
                found = emotion
                break
        normalized.append(found)

    acc = accuracy(normalized, dataset.references)

    return {"accuracy": acc["accuracy"]}

async def main():
    results = await test_emotion_classification(emotion_dataset)

    print("\n=== Emotion Classification Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Zero-Shot Classification

```python
import asyncio
from benchwise import evaluate, Dataset

# Test zero-shot classification with novel categories
zero_shot_texts = [
    "The quantum computer achieved new performance records",
    "Local restaurant wins culinary award",
    "New archaeological discovery sheds light on ancient civilization"
]

# Categories not explicitly trained on
zero_shot_labels = ["quantum_computing", "culinary", "archaeology"]

zero_shot_data = [
    {"text": text, "label": label}
    for text, label in zip(zero_shot_texts, zero_shot_labels)
]

zero_shot_dataset = Dataset(name="zero_shot", data=zero_shot_data)

@evaluate("gpt-4", "claude-3-opus")
async def test_zero_shot(model, dataset):
    prompts = [
        f"Classify into the most specific category possible: '{item['text']}'"
        for item in dataset.data
    ]

    responses = await model.generate(prompts, temperature=0)

    # Check if model identifies relevant categories
    correct = 0
    for response, item in zip(responses, dataset.data):
        response_lower = response.lower()
        expected = item["label"].replace("_", " ")
        if expected in response_lower or any(word in response_lower for word in expected.split("_")):
            correct += 1

    return {
        "zero_shot_accuracy": correct / len(responses),
        "successful_classifications": correct
    }

async def main():
    results = await test_zero_shot(zero_shot_dataset)

    print("\n=== Zero-Shot Classification Results ===")
    for result in results:
        if result.success:
            print(f"\n{result.model_name}:")
            print(f"  Zero-Shot Accuracy: {result.result['zero_shot_accuracy']:.2%}")
            print(f"  Successful Classifications: {result.result['successful_classifications']}")
        else:
            print(f"\n{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
```

## Saving and Analyzing Results

```python
import asyncio
from benchwise import (
    evaluate,
    benchmark,
    create_classification_dataset,
    accuracy,
    save_results,
    BenchmarkResult,
    ResultsAnalyzer
)

# Create sentiment classification dataset
texts = [
    "This product is amazing! I love it!",
    "Terrible experience, very disappointed.",
    "It's okay, nothing special.",
    "Best purchase I've ever made!",
    "Waste of money, do not recommend.",
    "Pretty good, meets expectations."
]

labels = [
    "positive",
    "negative",
    "neutral",
    "positive",
    "negative",
    "neutral"
]

sentiment_dataset = create_classification_dataset(
    texts=texts,
    labels=labels,
    name="sentiment_analysis"
)

@benchmark("Sentiment Analysis", "Product review sentiment classification")
@evaluate("gpt-3.5-turbo", "claude-3-haiku", "gemini-pro")
async def test_sentiment_classification(model, dataset):
    prompts = [f"Classify the sentiment as positive, negative, or neutral: '{text}'"
               for text in dataset.prompts]
    responses = await model.generate(prompts, temperature=0)
    normalized_responses = []
    for response in responses:
        response_lower = response.lower()
        if "positive" in response_lower:
            normalized_responses.append("positive")
        elif "negative" in response_lower:
            normalized_responses.append("negative")
        elif "neutral" in response_lower:
            normalized_responses.append("neutral")
        else:
            normalized_responses.append("unknown")
    acc = accuracy(normalized_responses, dataset.references)
    return {
        "accuracy": acc["accuracy"],
        "total_classified": len(normalized_responses),
        "unknown_responses": normalized_responses.count("unknown")
    }

async def run_complete_classification():
    results = await test_sentiment_classification(sentiment_dataset)

    # Create benchmark
    benchmark = BenchmarkResult(
        "Classification Benchmark",
        metadata={"task": "classification", "date": "2024-11-16"}
    )

    for result in results:
        benchmark.add_result(result)

    # Save results
    save_results(benchmark, "classification_results.json", format="json")
    save_results(benchmark, "classification_report.md", format="markdown")

    # Analyze
    comparison = benchmark.compare_models("accuracy")
    print(f"\nBest model: {comparison['best_model']}")
    print(f"Best accuracy: {comparison['best_score']:.2%}")

asyncio.run(run_complete_classification())
```

## Next Steps

- [Multi-Model Comparison](./multi-model-comparison.md) - Compare across all models
- [Question Answering](./question-answering.md) - QA evaluation examples
- [Metrics Guide](../guides/metrics.md) - Learn about classification metrics

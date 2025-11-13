import asyncio
from benchwise import evaluate, benchmark, create_qa_dataset, accuracy, semantic_similarity

# Create your dataset
qa_dataset = create_qa_dataset(
    questions=[
        "What is the capital of Japan?",
        "Who wrote '1984'?",
        "What is the speed of light?",
        "Explain photosynthesis in one sentence.",
        "What causes rainbows?"
    ],
    answers=[
        "Tokyo",
        "George Orwell",
        "299,792,458 meters per second",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Rainbows are caused by light refraction and reflection in water droplets."
    ],
    name="general_knowledge_qa"
)

@benchmark("General Knowledge QA", "Tests basic factual knowledge")
@evaluate("gpt-3.5-turbo", "gemini-2.5-flash-lite")
async def test_general_knowledge(model, dataset):
    responses = await model.generate(dataset.prompts)
    # Multiple metrics for comprehensive evaluation
    acc = accuracy(responses, dataset.references)
    similarity = semantic_similarity(responses, dataset.references)

    return {
        "accuracy": acc["accuracy"],
        "semantic_similarity": similarity["mean_similarity"],
        "total_questions": len(responses)
    }

# Run the evaluation
async def main():
    results = await test_general_knowledge(qa_dataset)

    print("\\n=== General Knowledge QA Results ===")
    for result in results:
        if result.success:
            print(f"{result.model_name}:")
            print(f"  Accuracy: {result.result['accuracy']:.2%}")
            print(f"  Similarity: {result.result['semantic_similarity']:.3f}")
        else:
            print(f"{result.model_name}: FAILED - {result.error}")

asyncio.run(main())
---
sidebar_position: 4
---

# Models

Understand how to work with different LLM providers in Benchwise.

## Supported Providers

Benchwise supports multiple LLM providers through a unified interface.

### OpenAI

```python
from benchwise import evaluate

@evaluate("gpt-4", "gpt-3.5-turbo", "gpt-4-turbo")
async def test_openai(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### Anthropic

```python
@evaluate("claude-3-opus", "claude-3-sonnet", "claude-3-5-haiku-20241022")
async def test_anthropic(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### Google

```python
@evaluate("gemini-pro", "gemini-1.5-pro")
async def test_google(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### HuggingFace

```python
@evaluate("microsoft/DialoGPT-medium", "gpt2")
async def test_huggingface(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### Mock Adapter

For testing without API calls:

```python
@evaluate("mock-test")
async def test_with_mock(model, dataset):
    responses = await model.generate(dataset.prompts)
    # Mock returns simple responses for testing
    return {"responses": responses}
```

## Model Interface

All model adapters provide a consistent async interface:

### Generate Text

```python
@evaluate("gpt-4")
async def generate_example(model, dataset):
    # Basic generation
    responses = await model.generate(dataset.prompts)

    # With parameters
    responses = await model.generate(
        dataset.prompts,
        temperature=0.7,
        max_tokens=500,
        top_p=0.9
    )

    return {"responses": responses}
```

### Get Token Count

```python
@evaluate("gpt-4")
async def token_counting(model, dataset):
    prompt = "How many tokens is this?"

    # Count tokens
    token_count = model.get_token_count(prompt)
    print(f"Tokens: {token_count}")

    responses = await model.generate([prompt])
    return {"tokens": token_count}
```

### Estimate Costs

```python
@evaluate("gpt-4")
async def cost_estimation(model, dataset):
    # Estimate costs before running
    input_tokens = sum(model.get_token_count(p) for p in dataset.prompts)
    output_tokens = 500  # Estimated

    cost = model.get_cost_estimate(input_tokens, output_tokens)
    print(f"Estimated cost: ${cost:.2f}")

    if cost > 10.0:
        raise ValueError("Evaluation too expensive!")

    responses = await model.generate(dataset.prompts)
    return {"cost": cost}
```

## Model Configuration

### Temperature

Control randomness in responses:

```python
# Deterministic (temperature=0)
@evaluate("gpt-4", temperature=0)
async def deterministic_test(model, dataset):
    # More consistent, factual responses
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

# Creative (temperature=0.9)
@evaluate("gpt-4", temperature=0.9)
async def creative_test(model, dataset):
    # More varied, creative responses
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### Max Tokens

Limit response length:

```python
@evaluate("gpt-4", max_tokens=100)
async def short_responses(model, dataset):
    # Responses limited to 100 tokens
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

### Top-p (Nucleus Sampling)

```python
@evaluate("gpt-4", top_p=0.9)
async def nucleus_sampling(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Using Model Adapters Directly

```python
from benchwise.models import get_model_adapter
import asyncio

async def direct_usage():
    # Get adapter
    adapter = get_model_adapter("gpt-4")

    # Generate
    responses = await adapter.generate(
        ["What is AI?"],
        temperature=0.7,
        max_tokens=200
    )

    print(responses[0])

asyncio.run(direct_usage())
```

## Model Selection Best Practices

### 1. Task-Appropriate Models

```python
# For complex reasoning - use larger models
@evaluate("gpt-4", "claude-3-opus")
async def complex_reasoning(model, dataset):
    pass

# For simple tasks - use smaller, faster models
@evaluate("gpt-3.5-turbo", "claude-3-haiku")
async def simple_tasks(model, dataset):
    pass
```

### 2. Cost-Performance Trade-offs

```python
# High-accuracy tasks
@evaluate("gpt-4", "claude-3-opus", temperature=0)
async def high_accuracy(model, dataset):
    pass

# Cost-effective bulk processing
@evaluate("gpt-3.5-turbo", "claude-3-haiku")
async def bulk_processing(model, dataset):
    pass
```

### 3. Provider-Specific Features

```python
# OpenAI function calling
@evaluate("gpt-4")
async def with_functions(model, dataset):
    # Use OpenAI-specific features
    pass

# Anthropic extended context
@evaluate("claude-3-opus")
async def long_context(model, dataset):
    # Leverage Claude's long context window
    pass
```

## Error Handling

Models automatically handle errors:

```python
@evaluate("gpt-4", "potentially-unavailable-model")
async def with_error_handling(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

results = asyncio.run(with_error_handling(dataset))

# Check for errors
for result in results:
    if not result.success:
        print(f"Error: {result.error}")
```

## Rate Limiting

Benchwise handles rate limiting automatically:

```python
@evaluate("gpt-4")
async def large_batch(model, dataset):
    # Automatically handles rate limits
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Next Steps

- [Evaluation Guide](./evaluation.md) - Learn evaluation patterns
- [Results Guide](./results.md) - Analyze results
- [API Reference](../api/models/model-adapter.md) - Model adapter API

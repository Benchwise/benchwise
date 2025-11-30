---
sidebar_position: 2
---

# OpenAIAdapter

Adapter for OpenAI models (GPT-4o, GPT-4, etc.).

## Supported Models

### GPT-4.5 Series
- `gpt-4.5` - Advanced reasoning model with emotional intelligence and improved factual accuracy.
- `gpt-4.5-mini` - A smaller, more cost-efficient version of GPT-4.5.

### GPT-4.1 Series
- `gpt-4.1` - Flagship model with 1-million-token context window, strong coding performance.
- `gpt-4.1-mini` - Smaller variant of GPT-4.1.
- `gpt-4.1-nano` - Ultra-small, highly optimized variant.

### o1 Series (Reasoning Models)
- `o1` - Advanced reasoning model (December 2024)
- `o1-mini` - Faster reasoning model
- `o1-preview` - Preview of O1 capabilities

### GPT-4o Series (Recommended)
- `gpt-4o` - Most capable multimodal model (May 2024), real-time text, audio, and image processing.
- `gpt-4o-2024-11-20` - Latest GPT-4o snapshot
- `gpt-4o-2024-08-06` - August 2024 snapshot
- `gpt-4o-2024-05-13` - Initial release snapshot
- `gpt-4o-mini` - Fast, affordable multimodal model with 128K context window.
- `gpt-4o-mini-2024-07-18` - Initial mini release

### GPT-4 Turbo Series
- `gpt-4-turbo` - Latest GPT-4 Turbo with vision
- `gpt-4-turbo-2024-04-09` - April 2024 snapshot
- `gpt-4-turbo-preview` - Preview version
- `gpt-4-0125-preview` - January 2024 preview

### GPT-4 Series
- `gpt-4` - Base GPT-4 model (March 2023)
- `gpt-4-0613` - June 2023 snapshot

### GPT-3.5 Series
- `gpt-3.5-turbo` - Cost-effective model
- `gpt-3.5-turbo-0125` - January 2024 snapshot

**Note:** Model availability and naming may change. Refer to [OpenAI's model documentation](https://platform.openai.com/docs/models) for the latest information.

## Usage

```python
from benchwise import evaluate

@evaluate("gpt-3.5-turbo", "gemini-2.5-flash", temperature=0.7)
async def test_openai(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Authentication

Set environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

## See Also

- [ModelAdapter](./model-adapter.md)
- [Models Guide](../../guides/models.md)

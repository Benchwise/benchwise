---
sidebar_position: 2
---

# OpenAIAdapter

Adapter for OpenAI models (GPT-4o, GPT-4, etc.).

## Supported Models

### GPT-4o Series (Recommended)
- `gpt-4o` - Most capable multimodal model
- `gpt-4o-mini` - Fast and affordable model for everyday tasks

### GPT-4 Turbo Series
- `gpt-4-turbo` - Latest GPT-4 Turbo
- `gpt-4-turbo-preview` - Preview version
- `gpt-4-0125-preview` - Latest preview snapshot
- `gpt-4-1106-preview` - November 2023 snapshot

### GPT-4 Series
- `gpt-4` - Base GPT-4 model
- `gpt-4-0613` - June 2023 snapshot

### GPT-3.5 Series (Legacy)
- `gpt-3.5-turbo` - Legacy model (consider using gpt-4o-mini instead)
- `gpt-3.5-turbo-0125` - Latest snapshot

**Note:** Model availability and naming may change. Refer to [OpenAI's model documentation](https://platform.openai.com/docs/models) for the latest information.

## Usage

```python
from benchwise import evaluate

@evaluate("gpt-4o", "gpt-4o-mini", temperature=0.7)
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

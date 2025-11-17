---
sidebar_position: 2
---

# OpenAIAdapter

Adapter for OpenAI models (GPT-3.5, GPT-4, etc.).

## Supported Models

- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Usage

```python
from benchwise import evaluate

@evaluate("gpt-4", "gpt-3.5-turbo", temperature=0.7)
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

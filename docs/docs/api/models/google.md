---
sidebar_position: 4
---

# GoogleAdapter

Adapter for Google Gemini models.

## Supported Models

- `gemini-pro`
- `gemini-1.5-pro`

## Usage

```python
from benchwise import evaluate

@evaluate("gemini-pro", "gemini-1.5-pro")
async def test_gemini(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Authentication

Set environment variable:
```bash
export GOOGLE_API_KEY="your-api-key"
```

## See Also

- [ModelAdapter](./model-adapter.md)
- [Models Guide](../../guides/models.md)

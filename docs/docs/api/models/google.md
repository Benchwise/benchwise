---
sidebar_position: 4
---

# GoogleAdapter

Adapter for Google Gemini models.

## Supported Models

### Gemini 2.5 Series (Deprecated as of November 2025)
- `gemini-2.5-pro` - Advanced reasoning model with multimodal capabilities (June 2025).
- `gemini-2.5-flash` - Optimized for speed and cost-efficiency (June 2025).
- `gemini-2.5-flash-lite` - Ultra-fast, lightweight, and cost-effective (June 2025).



## Usage

```python
from benchwise import evaluate

@evaluate("gpt-3.5-turbo", "gemini-2.5-flash")
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

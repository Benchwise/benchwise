---
sidebar_position: 3
---

# AnthropicAdapter

Adapter for Anthropic Claude models.

## Supported Models

- `claude-3-opus`
- `claude-3-sonnet`
- `claude-3-haiku`
- `claude-3-5-haiku-20241022`

## Usage

```python
from benchwise import evaluate

@evaluate("claude-3-opus", "claude-3-sonnet")
async def test_claude(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## Authentication

Set environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

## See Also

- [ModelAdapter](./model-adapter.md)
- [Models Guide](../../guides/models.md)

---
sidebar_position: 3
---

# AnthropicAdapter

Adapter for Anthropic Claude models.

## Supported Models

Current Anthropic API models with canonical versioned IDs:

- `claude-opus-4-20250514` - Claude Opus 4 (latest)
- `claude-opus-4-1-20250805` - Claude Opus 4.1
- `claude-sonnet-4-20250514` - Claude Sonnet 4 (latest)
- `claude-3-7-sonnet-20250219` - Claude 3.7 Sonnet
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet (October 2024)
- `claude-3-5-sonnet-20240620` - Claude 3.5 Sonnet (June 2024)
- `claude-3-5-haiku-20241022` - Claude 3.5 Haiku
- `claude-3-haiku-20240307` - Claude 3 Haiku
- `claude-3-opus-20240229` - Claude 3 Opus

**Note:** While the adapter also supports generic aliases like `claude-3-opus`, `claude-3-sonnet`, and `claude-3-haiku` for backward compatibility, it's recommended to use the canonical versioned model IDs above for consistent, reproducible results.

## Usage

```python
from benchwise import evaluate

@evaluate("gpt-3.5-turbo", "gemini-2.5-flash")
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

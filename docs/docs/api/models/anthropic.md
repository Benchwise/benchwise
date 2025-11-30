---
sidebar_position: 3
---

# AnthropicAdapter

Adapter for Anthropic Claude models.

## Supported Models

### Claude 4.5 Series (Latest)
- `claude-sonnet-4.5` - Strong model for coding, agents, and computer use, balancing performance and cost (September 2025). Features 200k context window and 64k output tokens.
- `claude-haiku-4.5` - Optimized for low latency and cost, suitable for real-time assistants (October 2025).

### Claude 3.5 Series
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet (October 2024, latest)
- `claude-3-5-sonnet-20240620` - Claude 3.5 Sonnet (June 2024)
- `claude-3-5-haiku-20241022` - Claude 3.5 Haiku (October 2024)

### Claude 3 Series
- `claude-3-opus-20240229` - Most capable Claude 3 model
- `claude-3-sonnet-20240229` - Balanced performance and speed
- `claude-3-haiku-20240307` - Fastest and most compact

## Usage

```python
from benchwise import evaluate

@evaluate("claude-haiku-4.5", "claude-3-5-sonnet-20241022")  
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

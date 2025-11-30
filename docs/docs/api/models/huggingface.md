---
sidebar_position: 5
---

# HuggingFaceAdapter

Adapter for HuggingFace models.

## Supported Models

Any HuggingFace model, e.g.:
- `microsoft/DialoGPT-medium`
- `gpt2`
- `facebook/bart-large`

## Usage

```python
from benchwise import evaluate

@evaluate("microsoft/DialoGPT-medium")
async def test_hf(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}
```

## See Also

- [ModelAdapter](./model-adapter.md)
- [Models Guide](../../guides/models.md)

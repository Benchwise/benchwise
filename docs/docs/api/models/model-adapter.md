---
sidebar_position: 1
---

# ModelAdapter

Abstract base class for all model adapters.

## Interface

```python
class ModelAdapter(ABC):
    @abstractmethod
    async def generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate responses for prompts"""
        ...

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens in text"""
        ...

    @abstractmethod
    def get_cost_estimate(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for token usage"""
        ...
```

## Factory Function

```python
def get_model_adapter(model_name: str, config: Optional[Dict[str, Any]] = None) -> ModelAdapter:
    ...
```

Automatically selects the appropriate adapter based on model name prefix:
- `gpt-*` → OpenAIAdapter
- `claude-*` → AnthropicAdapter
- `gemini-*` → GoogleAdapter
- `mock-*` → MockAdapter
- Others → HuggingFaceAdapter

## Usage

```python
from benchwise.models import get_model_adapter

adapter = get_model_adapter("gpt-4")
responses = await adapter.generate(["What is AI?"])
```

## See Also

- [OpenAI](./openai.md)
- [Anthropic](./anthropic.md)
- [Google](./google.md)
- [HuggingFace](./huggingface.md)

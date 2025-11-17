---
sidebar_position: 5
---

# Logging

Configure logging for debugging and monitoring.

## Enable Debug Logging

```python
from benchwise import configure_benchwise

configure_benchwise(debug=True)
```

## Environment Variable

```bash
export BENCHWISE_DEBUG="true"
```

## Custom Logging

```python
import logging

# Configure Python logging
logging.basicConfig(level=logging.DEBUG)

# Benchwise will use this configuration
```

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages
- **ERROR**: Error messages

## Example Output

```
INFO: Starting evaluation for gpt-4
DEBUG: Generating responses for 10 prompts
DEBUG: Received 10 responses in 5.2s
INFO: Evaluation completed successfully
```

## See Also

- [Configuration](./configuration.md)
- [Error Handling](./error-handling.md)

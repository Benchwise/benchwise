# Changelog

All notable changes to Benchwise will be documented in this file.


## [0.1.0a3]

### Added
- **Logging System**: Added comprehensive logging throughout the library
  - New `setup_logging()` function for easy configuration
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - File logging support
  - All operations now log important events
- **Request ID Tracking**: Every API request now includes a unique request ID
  - Improves debugging and support
  - Request IDs included in error messages
- **Custom Exceptions**: New exception hierarchy for better error handling
  - `BenchwiseError` - Base exception
  - `AuthenticationError` - Authentication failures
  - `RateLimitError` - Rate limit exceeded (includes retry_after)
  - `ValidationError` - Input validation failures
  - `NetworkError` - Network request failures
  - `ConfigurationError` - Configuration issues
  - `DatasetError` - Dataset operation failures
  - `ModelError` - Model operation failures
  - `MetricError` - Metric calculation failures
- **Context-Safe Client**: Replaced global client with context-local storage
  - Thread-safe and async-safe
  - Better isolation in concurrent scenarios
  - Automatic cleanup

### Changed
- **Improved @evaluate Decorator**: Now properly handles both sync and async test functions
  - Automatically detects if test function is async or sync
  - No more confusion about when to use `asyncio.run()`
  - Better error messages and logging
- **Enhanced Client Error Messages**: All errors now include more context
  - Request IDs for debugging
  - Status codes where applicable
  - Retry information for rate limits
- **Better Async/Sync Handling**: Unified approach across all decorators
  - `@evaluate` works seamlessly with both patterns
  - Internal helper functions properly handle async/sync code paths

### Fixed
- Fixed global client state causing issues in multi-threaded environments
- Fixed async/sync confusion in decorators
- Improved error handling in API requests
- Better retry logic with exponential backoff

### WIP (Work In Progress)
- Simplified upload workflow (`upload_benchmark_result_simple()`)
  - Currently redirects to existing multi-step flow
  - Will be completed in next release (0.1.0b1)
  - Goal: Single API call instead of 4+ calls

### Documentation
- Added `examples/a3_improvements.py` demonstrating new features
- Updated docstrings with logging and exception information

## [0.1.0a2]

### Added
- Initial alpha release
- Core evaluation framework with @evaluate and @benchmark decorators
- Multi-provider support (OpenAI, Anthropic, Google, HuggingFace)
- Comprehensive metrics (ROUGE, BLEU, BERTScore, etc.)
- Dataset management
- Results caching and analysis
- CLI interface
- API client with offline queue

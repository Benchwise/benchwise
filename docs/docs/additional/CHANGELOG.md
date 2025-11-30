# Changelog

All notable changes to Benchwise will be documented in this file.


## [0.1.0a2](Under development)

### Added
- **Logging System**: Added comprehensive logging throughout the library
  - New `setup_logging()` function for easy configuration
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - File logging support
  - All operations now log important events
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

### Changed
- **Improved @evaluate Decorator**: 
  - Better error messages and logging
- **Enhanced Client Error Messages**: All errors now include more context
  - Status codes where applicable
  - Retry information for rate limits

### Fixed
- Fixed global client state causing issues in multi-threaded environments
- Fixed async/sync confusion in decorators
- Improved error handling in API requests
- Better retry logic with exponential backoff

### WIP (Work In Progress)
- Improve Type Annotations and Static Type Checking

### Documentation
- Added `examples/a3_improvements.py` demonstrating new features
- Updated docstrings with logging and exception information

## [0.1.0a1]

### Added
- Initial alpha release
- Core evaluation framework with @evaluate and @benchmark decorators
- Multi-provider support (OpenAI, Anthropic, Google, HuggingFace)
- Comprehensive metrics (ROUGE, BLEU, BERTScore, etc.)
- Dataset management
- Results caching and analysis
- CLI interface
- API client with offline queue

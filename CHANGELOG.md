# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-14

### Added
- Full Anthropic Messages API to OpenAI Chat Completions translation (streaming and non-streaming)
- Tool use support (definitions, tool_use blocks, tool_result blocks, error handling)
- Image content block passthrough (base64)
- Multi-backend routing with model-to-URL mappings
- Semaphore-based request queuing with configurable concurrency limits
- In-memory LRU prompt cache with TTL and optional disk persistence
- Request/response recording and replay (JSONL format)
- Token budget tracking and enforcement with per-model breakdown
- Structured JSONL analytics logging
- Live terminal dashboard with real-time metrics
- Auto model discovery from backend `/v1/models` endpoint
- Hot-swap configuration via `/admin/reload` endpoint
- Cross-platform process supervisor with auto-restart and health checks
- Idempotent install script with llama.cpp build and model download
- Comprehensive test suite (106 tests)
- Documentation: README, installation guide, usage guide, Claude Code setup guide

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-08

### Added
- Provider gateway: OpenAI Chat Completions is now the hub format, with two inbound dialects (Anthropic `/v1/messages`, OpenAI `/v1/chat/completions`) and two upstream dialects (`openai` passthrough, `anthropic` translated)
- `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`
- Provider registry (`src/proxy/providers.py`) with presets: openai, anthropic, gemini (OpenAI-compatible endpoint), perplexity, groq, together, mistral, openrouter, deepseek, xai, azure_openai, custom
- Per-request upstream selection via `X-Lana-Upstream-Provider`, `X-Lana-Upstream-Url`, `X-Lana-Upstream-Key`; config routing is unchanged when they are absent
- OpenAI to Anthropic translation (`src/proxy/translation/openai_to_anthropic/`): system hoisting, tool calls and tool results, images, stop reasons, usage, and Anthropic SSE to OpenAI SSE streaming with incremental tool call deltas and optional usage chunk
- Perplexity `citations` and `search_results` pass through unchanged
- Upstream security: keys never stored, logged, recorded or echoed; X-Lana headers never forwarded; https required except loopback; private and link-local addresses refused for non-custom presets; redirects not followed
- OpenAI-format error bodies for the OpenAI inbound dialect, with upstream status mapping
- Tests for providers, reverse translation and end-to-end gateway behavior against a local fake upstream (212 tests total)

### Changed
- Streaming responses now send `Connection: close` instead of `keep-alive`. SSE bodies have no Content-Length or chunked framing, so clients that read to end of stream previously waited forever.

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

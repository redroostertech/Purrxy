# Connecting Claude Code to purrxy

Run Claude Code against your local llama.cpp backend through purrxy.

## Prerequisites

1. purrxy installed (`bash install.sh`)
2. Backend + proxy running (`./start.sh`)
3. Claude Code installed (`npm install -g @anthropic-ai/claude-code` or however you installed it)

Verify the stack is healthy:

```bash
curl http://127.0.0.1:5304/health
# {"status": "ok"}
```

## Quick Start

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:5304 claude
```

That's it. Claude Code sends requests to purrxy, which translates them and forwards to your local llama.cpp.

## Make It Permanent

Add the environment variable to your shell config so you don't have to type it every time:

```bash
echo 'export ANTHROPIC_BASE_URL=http://127.0.0.1:5304' >> ~/.zshrc
source ~/.zshrc
```

Then just run:

```bash
claude
```

## API Key

Claude Code expects an API key. purrxy ignores authentication entirely, but Claude Code may still require the variable to be set. If you get an auth error:

```bash
export ANTHROPIC_API_KEY=not-needed
```

Or combine both:

```bash
echo 'export ANTHROPIC_BASE_URL=http://127.0.0.1:5304' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY=not-needed' >> ~/.zshrc
source ~/.zshrc
```

## Switching Between Local and Cloud

### Option A: Alias

```bash
# Add to ~/.zshrc
alias claude-local='ANTHROPIC_BASE_URL=http://127.0.0.1:5304 ANTHROPIC_API_KEY=not-needed claude'
```

Then:

```bash
claude           # Uses Anthropic cloud (if ANTHROPIC_BASE_URL is not set globally)
claude-local     # Uses local llama.cpp via purrxy
```

### Option B: Shell Function

```bash
# Add to ~/.zshrc
claude-local() {
    ANTHROPIC_BASE_URL=http://127.0.0.1:5304 \
    ANTHROPIC_API_KEY=not-needed \
    claude "$@"
}
```

This passes all arguments through, so `claude-local --help`, `claude-local -p "explain this"`, etc. all work.

### Option C: Toggle Script

```bash
# Add to ~/.zshrc
toggle-claude() {
    if [ -z "$ANTHROPIC_BASE_URL" ]; then
        export ANTHROPIC_BASE_URL=http://127.0.0.1:5304
        export ANTHROPIC_API_KEY=not-needed
        echo "Claude Code → LOCAL (purrxy)"
    else
        unset ANTHROPIC_BASE_URL
        unset ANTHROPIC_API_KEY
        echo "Claude Code → CLOUD (Anthropic)"
    fi
}
```

## Model Names

Claude Code sends model names like `claude-sonnet-4-20250514`. purrxy forwards these to the backend as-is by default — llama.cpp ignores the model name field and uses whatever model it has loaded.

If you want explicit model mapping:

```bash
purrxy --model-map claude-sonnet-4-20250514=qwen2.5-coder
```

This is cosmetic — it renames the model in the request before forwarding. The backend still uses its loaded model regardless.

## What Works

| Claude Code Feature | Status | Notes |
|---|---|---|
| Text conversations | Works | Single and multi-turn |
| Streaming responses | Works | Real-time token output |
| Tool use (file read/write/edit) | Works | Full tool call round-trips |
| System prompts | Works | Translated to system message |
| `/help`, `/clear`, etc. | Works | Client-side, no proxy involved |
| Image inputs | Translated | Sent as `[image: type]` placeholder — backend LLM can't process images |
| Thinking/extended thinking | Ignored | Logged as warning, stripped from request |

## Performance Expectations

Local inference is slower than Anthropic's cloud API. Expect:

- **First token latency:** 1–5 seconds (vs. ~0.5s cloud)
- **Token throughput:** 20–40 tokens/sec on M4 Pro with 14B model (vs. ~100+ cloud)
- **Context window:** Limited by your RAM and `CONTEXT_SIZE` setting (default 8192 tokens)

Claude Code works fine at these speeds — it just takes longer per response. The experience is best for shorter, focused tasks.

### Tips for Better Performance

- **Keep context small.** Use `/clear` between unrelated tasks to avoid accumulating large conversation history.
- **Use caching.** If you're iterating on the same prompt: `purrxy --cache --cache-ttl 600`
- **Increase context carefully.** Going from 8K to 16K doubles RAM pressure. See the [install guide](INSTALL.md#context-size-guidelines) for recommendations.

## Troubleshooting

### "Connection refused" or timeout

The proxy or backend isn't running:

```bash
# Check both services
curl http://127.0.0.1:5304/health    # Proxy
curl http://127.0.0.1:8080/health    # Backend

# Restart everything
./stop.sh && ./start.sh
```

### Claude Code hangs on first message

The backend is still loading the model. Large models take 10–60 seconds. Check:

```bash
cat ~/.config/purrxy/backend.log
```

Wait for the model to finish loading, then retry.

### "Overloaded" or 529 errors

The backend is busy with another request. The proxy queues requests by default (`--max-concurrent 1`). If the queue timeout expires (default 120s), you get a 529.

Options:
- Wait and retry
- Increase queue timeout: `purrxy --queue-timeout 300`
- Run backend with parallel slots: set `PARALLEL_SLOTS=2` in `.env` (uses more RAM)

### Garbled or incomplete tool use

The local model may not follow tool-use formatting as reliably as Claude. This is a model quality limitation, not a proxy issue. Tips:

- Use larger/better quantized models for tool-heavy workflows
- The 14B Q5_K_M model handles basic tool use well
- Complex multi-tool chains may need a larger model

### Reverting to Cloud

Remove or comment out the environment variables:

```bash
# In ~/.zshrc, remove or comment:
# export ANTHROPIC_BASE_URL=http://127.0.0.1:5304
# export ANTHROPIC_API_KEY=not-needed
source ~/.zshrc
```

Or just unset for the current session:

```bash
unset ANTHROPIC_BASE_URL
unset ANTHROPIC_API_KEY
claude
```

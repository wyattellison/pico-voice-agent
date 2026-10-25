# pico-voice-agent

LiveKit voice agent for talking to Pico live: faster-whisper (STT) ->
local Ollama (LLM) -> VoiceStudio/OmniVoice (TTS). Joins the
`pico-livekit` room as a participant using the `livekit-agents` SDK.

## Why this isn't deployed through the pico-services-controller

It needs `network_mode: host` to reach Ollama (`127.0.0.1:11434`) and
VoiceStudio (`127.0.0.1:3900`) -- both bound to loopback only, so
`host.docker.internal`/host-gateway traffic (which arrives from the
bridge IP, not loopback) can't reach either. The controller's
`validateComposeFile()` rejects `network_mode: host` for anything it
deploys, by design. So run this one directly:

```bash
cp .env.example .env   # fill in LIVEKIT_API_KEY/SECRET, OMNIVOICE_API_KEY
docker compose up -d --build
docker compose logs -f
```

## Things to verify before trusting this scaffold

1. **`TTS_PATH` in `agent.py`** (`/v1/tts`) is a guess -- check
   `VOICESTUDIO_URL/openapi.json` for the real synthesis route and
   request/response shape, and fix the `VoiceStudioTTS.synthesize()` call
   to match.
2. **`livekit-agents` API surface** -- written without being able to run
   it locally. Check `pip show livekit-agents` and its plugin docs for
   the actual `LLM`/`STT`/`TTS` base class method signatures (`chat()`,
   `_recognize_impl()`, `synthesize()`) if anything fails to import or
   subclass cleanly.
3. **Whisper performance/placement** -- defaults to `small` on CPU. The
   GPU (GTX 1080 Ti, 11GB) is already tight with Ollama's model plus
   VoiceStudio; only move Whisper to GPU if there's headroom left after
   both of those are loaded (check with `nvidia-smi` while a call is
   live).
4. **`OLLAMA_MODEL`** defaults to `qwen3:8b` (not the currently-installed
   `qwen3:14b`, which doesn't fit in VRAM alongside VoiceStudio+Whisper) --
   pull it on the tower (`ollama pull qwen3:8b`) before starting this.

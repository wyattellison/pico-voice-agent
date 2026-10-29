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

## Verification status (updated 2026-10-28)

This was written and fixed without ever running it -- no livekit-agents
or VoiceStudio instance was reachable from where the code was edited.
Everything below is "matches the real docs/source I could read," **not**
"confirmed working." Someone with tower shell access still needs to
actually run this and debug whatever the docs didn't cover.

### Fixed this pass, checked against real source

- **`livekit-agents` version pin was wrong.** `requirements.txt` said
  `>=0.12`, but the code uses 1.x-era classes (`Agent`, `AgentSession`,
  `LLMStream`, `ChunkedStream`, `AudioEmitter`, `ChatContext.items`).
  Bumped to `>=1.0`. If the tower's pip resolves something pre-1.0, this
  whole file breaks at import.
- **`LLM` plugin contract.** Confirmed against `livekit/agents` main
  (`livekit-agents/livekit/agents/llm/llm.py`): `LLM.chat()` is a plain
  (non-async) method returning an `LLMStream`; the real work happens in
  `LLMStream._run(self) -> None`, which pushes `ChatChunk(id=...,
  delta=ChoiceDelta(role=..., content=...))` onto `self._event_ch`. The
  previous version had `chat()` return a bare async-iterable and used
  `ChatChunk(choices=[...])`, which isn't a real field on `ChatChunk`
  (it's `delta`, singular) -- that would have failed at the first
  streamed token. `OllamaLLM`/`OllamaLLMStream` now subclass correctly.
- **`TTS` plugin contract.** Confirmed against `livekit/agents` main
  (`tts/tts.py`) and the real `elevenlabs` plugin as a reference impl:
  `TTS.synthesize()` is non-async and must return a `ChunkedStream`; the
  HTTP call belongs in `ChunkedStream._run(self, output_emitter)`, which
  calls `output_emitter.initialize(request_id=, sample_rate=,
  num_channels=, mime_type=)` then `.push(bytes)` / `.flush()`. The
  previous version had `synthesize()` directly `await` an HTTP call and
  return a `SynthesizedAudio` object, which doesn't match this contract
  at all. `VoiceStudioTTS`/`VoiceStudioChunkedStream` now follow the real
  pattern; `mime_type="audio/pcm"` matches `AudioEmitter`'s raw-PCM
  detection (`mt.startswith("audio/pcm")`).
- **`STT._recognize_impl()` signature** was already close, but the old
  code assumed `AudioBuffer` has a `.to_wav_bytes()` method -- it
  doesn't; `AudioBuffer = list[rtc.AudioFrame] | rtc.AudioFrame`. Now
  uses `livekit.agents.utils.merge_frames()` to collapse the buffer into
  one `AudioFrame`, resamples to 16kHz via `rtc.AudioResampler` if needed
  (confirmed `push()`/`flush()` both return `list[AudioFrame]`), and
  converts the int16 PCM to a float32 ndarray for `faster-whisper`
  (confirmed `WhisperModel.transcribe()` accepts a float32 numpy array
  directly -- no need to round-trip through a WAV file).
- **No official first-party `livekit-agents` plugin for faster-whisper
  exists** (checked 2026-10-28 -- only community wrappers like
  `taresh18/livekit-whisper`), so the custom `STT` subclass approach here
  is the right call, not a shortcut to revisit.
- **VoiceStudio's TTS route.** `TTS_PATH` was a pure guess before
  (`/v1/tts`). Now `/v1/audio/speech`, cross-confirmed from two
  independent sources: `debpalash/VoiceStudio` PR #2290 ("make /v1 audio
  routes work with the openai and Agents SDKs") and voicestudio.sh's
  published API reference. Both describe an OpenAI-compatible
  `POST /v1/audio/speech` taking `{model, voice, input, response_format,
  speed}` and returning raw audio bytes with a matching `Content-Type`,
  auth via `Authorization: Bearer <key>`. `response_format: "pcm"` is
  documented as 24kHz int16 LE mono, matching this file's
  `sample_rate=24000, num_channels=1`.

### Still unverified -- the live instance was never reachable

- VoiceStudio's actual behavior when hit for real: whether
  `OMNIVOICE_API_KEY` auth is enforced the way the docs describe, whether
  `voice: "default"` is a valid value out of the box, exact latency.
- Whether `LLM.__init__()`/base `rtc.EventEmitter.__init__()` take zero
  args -- inferred from the pattern other plugins follow, not read
  directly off that constructor's source.
- Whisper performance/placement -- defaults to `small` on CPU. The GPU
  (GTX 1080 Ti, 11GB) is already tight with Ollama's model plus
  VoiceStudio; only move Whisper to GPU if there's headroom left after
  both of those are loaded (check with `nvidia-smi` while a call is
  live).
- `OLLAMA_MODEL` defaults to `qwen3:8b` (not the currently-installed
  `qwen3:14b`, which doesn't fit in VRAM alongside VoiceStudio+Whisper) --
  pull it on the tower (`ollama pull qwen3:8b`) before starting this.
- General import/runtime errors this review wouldn't catch -- this was
  never actually executed against a real Python environment.

## Next step

Tower shell access needed: `git pull`, `docker compose up --build -d` in
this directory, then `docker compose logs -f` and fix whatever the docs
above didn't anticipate.

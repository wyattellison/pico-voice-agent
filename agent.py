"""
pico-voice-agent: joins a LiveKit room and runs the live voice loop --
faster-whisper (STT) -> Ollama (LLM) -> VoiceStudio/OmniVoice (TTS).

Updated 2026-10-28: the LLM/STT/TTS plugin wiring below was rewritten
against the real livekit-agents (1.x) base-class contracts and
VoiceStudio's real OpenAI-compatible /v1/audio/speech route, verified via
docs/source (not via actually running it -- no livekit-agents or
VoiceStudio instance was reachable from where this was written). See
README.md for the itemized verified-vs-still-guessed breakdown before
trusting this on the tower.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import numpy as np
from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, utils
from livekit.agents.llm import LLM, LLMStream, ChatContext, ChatChunk, ChoiceDelta
from livekit.agents.stt import STT, SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.tts import TTS, ChunkedStream, TTSCapabilities, AudioEmitter
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given, merge_frames
from livekit.plugins import silero

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
VOICESTUDIO_URL = os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:3900")
OMNIVOICE_API_KEY = os.environ.get("OMNIVOICE_API_KEY", "")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_SAMPLE_RATE = 16000  # faster-whisper expects 16kHz float32 for raw ndarray input

BASE_PERSONA_PROMPT = (
    "You are Pico, speaking with Wyatt over a live voice call. "
    "Keep replies short and conversational -- a sentence or two, not a "
    "written essay. No markdown, no lists; this is speech."
)

DIGEST_PATH = Path(__file__).parent / "context_digest.md"

# VoiceStudio's OpenAI-compatible synthesis route. Confirmed 2026-10-28
# against debpalash/VoiceStudio PR #2290 ("make /v1 audio routes work
# with the openai and Agents SDKs") and voicestudio.sh's published API
# docs -- both independently describe POST /v1/audio/speech taking
# {model, voice, input, response_format, speed} and returning raw audio
# bytes with a matching Content-Type, auth via `Authorization: Bearer
# <key>`. NOT confirmed against a live instance (unreachable from here).
TTS_PATH = "/v1/audio/speech"
TTS_MODEL = os.environ.get("VOICESTUDIO_MODEL", "tts-1")
TTS_VOICE = os.environ.get("VOICESTUDIO_VOICE", "default")


def build_system_prompt() -> str:
    """Reads context_digest.md fresh on every call (not baked in at agent
    startup) so calendar/project status drift doesn't go stale between
    deploys."""
    try:
        digest = DIGEST_PATH.read_text().strip()
    except OSError:
        digest = ""
    if digest:
        return f"{BASE_PERSONA_PROMPT}\n\n{digest}"
    return BASE_PERSONA_PROMPT


class OllamaLLM(LLM):
    """Thin LLM wrapper calling Ollama's native /api/chat directly
    (bypassing the OpenAI-compat shim) so we can pass think=False, per
    Qwen3's default reasoning preamble adding seconds of dead air.

    Real livekit-agents LLM contract (verified against livekit/agents
    main, 2026-10-28): `LLM.chat()` is a plain (non-async) method that
    returns an `LLMStream` instance; the actual request/response work
    happens in `LLMStream._run()`, which pushes `ChatChunk`s onto
    `self._event_ch`. The old scaffold had `chat()` return a bare
    async-iterable object and `ChatChunk(choices=[...])`, which doesn't
    match the real API (no `choices` field -- it's a single `delta`).
    """

    def __init__(self, system_prompt: str):
        super().__init__()
        self._system_prompt = system_prompt

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[object] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> "OllamaLLMStream":
        return OllamaLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            system_prompt=self._system_prompt,
        )


class OllamaLLMStream(LLMStream):
    def __init__(
        self,
        llm: OllamaLLM,
        *,
        chat_ctx: ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
        system_prompt: str,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._system_prompt = system_prompt

    async def _run(self) -> None:
        messages = [{"role": "system", "content": self._system_prompt}]
        for item in self._chat_ctx.items:
            # ChatContext.items can hold function calls/results/handoffs too;
            # Ollama's native /api/chat only wants plain role+content turns.
            if getattr(item, "type", None) != "message":
                continue
            text = item.text_content
            if text:
                messages.append({"role": item.role, "content": text})

        request_id = utils.shortuuid()
        timeout = httpx.Timeout(self._conn_options.timeout + 30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "keep_alive": -1,
                    "options": {"num_ctx": 4096},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        self._event_ch.send_nowait(
                            ChatChunk(
                                id=request_id,
                                delta=ChoiceDelta(role="assistant", content=content),
                            )
                        )
                    if chunk.get("done"):
                        break


class FasterWhisperSTT(STT):
    """No official first-party livekit-agents plugin for faster-whisper
    exists (checked 2026-10-28 -- only community wrappers, e.g.
    taresh18/livekit-whisper), so a custom wrapper against the STT base
    class is the right call, as the original scaffold assumed.

    `_recognize_impl()`'s signature below matches the real base class
    (confirmed against livekit/agents main); the main fix vs. the old
    scaffold is building a proper float32/16kHz ndarray for faster-whisper
    instead of assuming a `.to_wav_bytes()` method that doesn't exist on
    `AudioBuffer`.
    """

    def __init__(self):
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type="int8" if WHISPER_DEVICE == "cpu" else "float16",
        )

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        frame = merge_frames(buffer)

        if frame.sample_rate != WHISPER_SAMPLE_RATE:
            resampler = rtc.AudioResampler(
                frame.sample_rate, WHISPER_SAMPLE_RATE, num_channels=frame.num_channels
            )
            frames = resampler.push(frame) + resampler.flush()
            frame = merge_frames(frames)

        pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        lang = language if is_given(language) else "en"

        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, lambda: self._model.transcribe(pcm, language=lang)
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[SpeechData(text=text, language=lang)],
        )


class VoiceStudioTTS(TTS):
    """VoiceStudio's OpenAI-compatible /v1/audio/speech route -- see the
    TTS_PATH comment above for what's verified vs. not. Real
    livekit-agents TTS contract (confirmed against livekit/agents main
    and the elevenlabs plugin as a reference implementation):
    `TTS.synthesize()` is non-async and returns a `ChunkedStream`; the
    actual HTTP call happens in `ChunkedStream._run(output_emitter)`,
    which must call `output_emitter.initialize(...)` then
    `output_emitter.push(...)`/`.flush()` -- not just return audio bytes
    directly, which is what the old scaffold did.
    """

    def __init__(self):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "VoiceStudioChunkedStream":
        return VoiceStudioChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class VoiceStudioChunkedStream(ChunkedStream):
    async def _run(self, output_emitter: AudioEmitter) -> None:
        timeout = httpx.Timeout(self._conn_options.timeout + 20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{VOICESTUDIO_URL}{TTS_PATH}",
                headers={"Authorization": f"Bearer {OMNIVOICE_API_KEY}"},
                json={
                    "model": TTS_MODEL,
                    "voice": TTS_VOICE,
                    "input": self._input_text,
                    # "pcm" per VoiceStudio's PR #2290: resampled to 24kHz
                    # int16 LE mono -- matches sample_rate/num_channels above.
                    "response_format": "pcm",
                    "speed": 1.0,
                },
            )
            resp.raise_for_status()

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=24000,
                num_channels=1,
                mime_type="audio/pcm",
            )
            output_emitter.push(resp.content)
            output_emitter.flush()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=rtc.AutoSubscribe.AUDIO_ONLY)

    system_prompt = build_system_prompt()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=FasterWhisperSTT(),
        llm=OllamaLLM(system_prompt),
        tts=VoiceStudioTTS(),
    )

    agent = Agent(instructions=system_prompt)
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

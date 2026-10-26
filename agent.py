"""
pico-voice-agent: joins a LiveKit room and runs the live voice loop --
faster-whisper (STT) -> Ollama (LLM) -> VoiceStudio/OmniVoice (TTS).

Scaffold only -- written without the ability to run livekit-agents locally,
so verify the LLM/STT/TTS base-class method signatures against whatever
`livekit-agents` version actually installs (`pip show livekit-agents`,
check its plugin docs) before trusting this as-is. Known unknowns, confirm
on the tower:
  - VoiceStudio's real synthesis route/payload shape (TTS_PATH below is a
    guess -- check VOICESTUDIO_URL + "/openapi.json" and fix it).
  - Whether `faster-whisper` on CPU is fast enough for live turn-taking,
    or whether it needs a GPU slot (budget permitting, see VRAM note in
    voice-chat-pico.md memory).
"""

import asyncio
import os

import httpx
from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import LLM, ChatContext, ChatChunk, ChoiceDelta
from livekit.agents.stt import STT, SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.tts import TTS, SynthesizedAudio, TTSCapabilities
from livekit.agents.utils import AudioBuffer
from livekit.plugins import silero

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
VOICESTUDIO_URL = os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:3900")
OMNIVOICE_API_KEY = os.environ.get("OMNIVOICE_API_KEY", "")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

SYSTEM_PROMPT = (
    "You are Pico, speaking with Wyatt over a live voice call. "
    "Keep replies short and conversational -- a sentence or two, not a "
    "written essay. No markdown, no lists; this is speech."
)

# Guess -- confirm the real path against VoiceStudio's own /openapi.json
# on the tower before relying on this.
TTS_PATH = "/v1/tts"


class OllamaLLM(LLM):
    """Thin LLM wrapper calling Ollama's native /api/chat directly
    (bypassing the OpenAI-compat shim) so we can pass think=False, per
    Qwen3's default reasoning preamble adding seconds of dead air."""

    def chat(self, *, chat_ctx: ChatContext, **kwargs) -> "OllamaLLMStream":
        return OllamaLLMStream(self, chat_ctx=chat_ctx)


class OllamaLLMStream:
    def __init__(self, llm: OllamaLLM, chat_ctx: ChatContext):
        self._llm = llm
        self._chat_ctx = chat_ctx

    async def __aiter__(self):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in self._chat_ctx.items:
            role = getattr(item, "role", "user")
            text = getattr(item, "text_content", None) or str(item)
            messages.append({"role": role, "content": text})

        async with httpx.AsyncClient(timeout=30.0) as client:
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
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield ChatChunk(
                            choices=[ChoiceDelta(role="assistant", content=content)]
                        )
                    if chunk.get("done"):
                        break


class FasterWhisperSTT(STT):
    def __init__(self):
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type="int8" if WHISPER_DEVICE == "cpu" else "float16",
        )

    async def _recognize_impl(self, buffer: AudioBuffer, **kwargs) -> SpeechEvent:
        samples = buffer.to_wav_bytes() if hasattr(buffer, "to_wav_bytes") else bytes(buffer)
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, lambda: self._model.transcribe(samples, language="en")
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text=text, language="en")],
        )


class VoiceStudioTTS(TTS):
    def __init__(self):
        super().__init__(capabilities=TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1)

    async def synthesize(self, text: str, **kwargs) -> SynthesizedAudio:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{VOICESTUDIO_URL}{TTS_PATH}",
                headers={"Authorization": f"Bearer {OMNIVOICE_API_KEY}"},
                json={"text": text, "format": "pcm_s16le", "sample_rate": 24000},
            )
            resp.raise_for_status()
            return SynthesizedAudio(data=resp.content, sample_rate=24000, num_channels=1)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=rtc.AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=FasterWhisperSTT(),
        llm=OllamaLLM(),
        tts=VoiceStudioTTS(),
    )

    agent = Agent(instructions=SYSTEM_PROMPT)
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

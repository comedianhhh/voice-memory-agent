"""Smoke test without a microphone: drive each plugin the agent uses, through LiveKit's own classes.

TTS -> audio frames, the same audio -> STT -> text (round trip), and one LLM turn via AgentSession.
Run: uv run smoke_test.py
"""

from __future__ import annotations

import asyncio
import time

from livekit import rtc
from livekit.agents import AgentSession

import agent as A

PHRASE = "Remind me that my favourite drink is a matcha latte."


async def main() -> None:
    tts = A.build_tts()
    t0 = time.perf_counter()
    frames, first = [], None
    async for ev in tts.synthesize(PHRASE):
        if first is None:
            first = time.perf_counter() - t0
        frames.append(ev.frame)
    audio = rtc.combine_audio_frames(frames)
    print(f"TTS  first audio {first*1000:.0f} ms, {audio.duration:.2f}s of audio")

    stt = A.openai.STT(base_url=A.SPEECH_URL, api_key="local", model="whisper-1", language="en")
    t0 = time.perf_counter()
    ev = await stt.recognize(audio)
    text = ev.alternatives[0].text
    print(f"STT  {(time.perf_counter()-t0)*1000:.0f} ms -> {text!r}")
    assert "matcha" in text.lower(), text

    llm = A.openai.LLM(base_url=A.OLLAMA_URL, api_key="ollama", model=A.LLM_MODEL,
                       reasoning_effort="none")
    async with AgentSession(llm=llm) as session:
        await session.start(A.Assistant())
        result = await session.run(user_input="Hi! What can you do?")
        reply = result.expect.next_event().is_message(role="assistant")
        print(f"LLM  -> {reply.event().item.text_content!r}")
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())

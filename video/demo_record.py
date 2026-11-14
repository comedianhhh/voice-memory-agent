"""Record a real two-session demo through the agent, for the video (no camera, no microphone).

A synthesized "user" (Kokoro voice am_michael) speaks into the real pipeline, as in bench.py:
  session 1: tells the agent two facts      -> the agent stores them (write_memory)
  session 2: a NEW session asks for them    -> the agent answers from memory (prefetch search)
Everything the agent says, the tools it calls and the latency it measured are recorded as they
happened. Nothing is scripted on the agent's side.

Output in video/demo/: timeline.json plus one WAV per utterance (user_*.wav 16 kHz, agent_*.wav 24 kHz).
Run:  uv run video/demo_record.py        (needs speech server, Ollama, memory API)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent as A  # noqa: E402
import bench as B  # noqa: E402
from livekit.agents.testing import fake_job_context  # noqa: E402
from livekit.plugins import silero  # noqa: E402

OUT = Path(__file__).parent / "demo"
MODE = "prefetch"  # the mode the benchmark recommends
SESSIONS = [
    [("u1", "Hi! Please remember that my dog is called Biscuit, and I'm flying to Lisbon next month.")],
    [("u2", "Hey, where am I flying next month, and what's my dog's name?")],
]
CHECKS = {"u1": lambda turn: "write_memory" in turn["tools"],
          "u2": lambda turn: "lisbon" in turn["reply"].lower() and "biscuit" in turn["reply"].lower()}


def save_wav(path: Path, pcm: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())


class RecordingSink(B.RealtimeSink):
    """RealtimeSink that also keeps the agent's audio and when each reply started."""

    def __init__(self, clock) -> None:
        super().__init__()
        self.clock = clock
        self.segments: list[dict] = []

    async def capture_frame(self, frame) -> None:
        if not self._dur:
            self.segments.append({"t": self.clock(), "pcm": []})
        self.segments[-1]["pcm"].append(np.frombuffer(frame.data, dtype=np.int16).copy())
        await super().capture_frame(frame)


async def run_session(idx: int, turns, vad, clock, timeline: list) -> None:
    session = A.build_session(vad)
    agent = A.MemoryAgent(MODE)
    mic, sink = B.ScriptedMic(), RecordingSink(clock)
    session.input.audio, session.output.audio = mic, sink
    cur: dict = {}

    @session.on("function_tools_executed")
    def _tools(ev) -> None:
        cur.setdefault("tools", []).extend(c.name for c in ev.function_calls)
        cur.setdefault("tool_args", []).extend(c.arguments for c in ev.function_calls)

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        item = ev.item
        if getattr(item, "type", "message") != "message":
            return
        if item.role == "user":
            cur["heard"] = item.text_content
            cur["stopped"] = (item.metrics or {}).get("stopped_speaking_at")
        elif item.role == "assistant":
            m = item.metrics or {}
            cur["reply"] = item.text_content or ""
            cur["e2e"] = m.get("e2e_latency")
            cur["llm_ttft"] = m.get("llm_node_ttft")
            cur["tts_ttfb"] = m.get("tts_node_ttfb")
            cur["end_of_turn"] = None

    await session.start(agent=agent, record=False)
    await agent._warmup_task  # what a live user's "hello, let me think" covers; see MemoryAgent.warm_up
    await asyncio.sleep(1.0)
    timeline.append({"type": "session", "n": idx + 1, "t": clock()})
    try:
        for uid, text in turns:
            cur.clear()
            pcm = B.prompt_audio(f"demo-{uid}", text)
            save_wav(OUT / f"user_{uid}.wav", pcm, B.MIC_RATE)
            t_user = clock()
            waiter = asyncio.create_task(B.wait_reply_done(session))
            await mic.say(pcm)
            await waiter
            seg = sink.segments[-1]
            apcm = np.concatenate(seg["pcm"])
            save_wav(OUT / f"agent_{uid}.wav", apcm, 24000)
            turn = {
                "type": "turn", "id": uid, "session": idx + 1,
                "said": text, "heard": cur.get("heard", ""), "reply": cur.get("reply", ""),
                "tools": cur.get("tools", []), "tool_args": cur.get("tool_args", []),
                "memory_hits": getattr(agent, "last_prefetch_hits", None),
                "prefetch_ms": agent.last_prefetch_ms,
                "e2e": cur.get("e2e"), "llm_ttft": cur.get("llm_ttft"), "tts_ttfb": cur.get("tts_ttfb"),
                "user_t": t_user, "user_dur": len(pcm) / B.MIC_RATE,
                "agent_t": seg["t"], "agent_dur": len(apcm) / 24000,
            }
            timeline.append(turn)
            ok = CHECKS[uid](turn)
            print(f"  [{uid}] heard={turn['heard']!r}\n        tools={turn['tools']} reply={turn['reply']!r}"
                  f"\n        e2e={turn['e2e']} ok={ok}", flush=True)
            if not ok:
                raise RuntimeError(f"take failed at {uid}")
            await asyncio.sleep(0.8)
    finally:
        await session.aclose()


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vad = silero.VAD.load()
    for take in range(1, 4):
        os.environ["MEMORY_EMAIL"], os.environ["MEMORY_PASSWORD"] = B.fresh_seeded_user(seed=[])  # empty memory: it must learn the facts in session 1
        start = time.perf_counter()
        clock = lambda: round(time.perf_counter() - start, 3)  # noqa: E731
        timeline: list = []
        try:
            with fake_job_context(inference_executor=B.LocalInference()):
                for i, turns in enumerate(SESSIONS):
                    await run_session(i, turns, vad, clock, timeline)
        except RuntimeError as e:
            print(f"take {take}: {e}, retrying with a fresh account", flush=True)
            continue
        (OUT / "timeline.json").write_text(json.dumps({"mode": MODE, "llm": A.LLM_MODEL, "take": take,
                                                      "events": timeline}, indent=2), encoding="utf-8")
        print(f"take {take} OK -> {OUT / 'timeline.json'}")
        return
    raise SystemExit("3 takes failed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())

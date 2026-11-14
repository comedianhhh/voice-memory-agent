"""Voice latency benchmark, no microphone: scripted spoken questions go through the real pipeline.

A fake mic plays pre-synthesized questions into AgentSession in real time (10 ms frames, silence
in between), so VAD, end-of-turn detection, STT, LLM, memory and TTS all run exactly as in a live
call. A fake speaker "plays" the reply in real time too, so turn-taking is realistic. Every reply's
per-stage latency is appended to data/bench.jsonl by agent.turn_logger.

Each memory mode gets a fresh local test user seeded with the same facts, then the same script.

Run:  uv run bench.py                      # all modes, 1 pass of the script
      uv run bench.py --modes tool prefetch --passes 3
Needs: speech server, Ollama, memory-service API (start.ps1 starts all three).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import AgentSession, io
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.testing import fake_job_context
from livekit.plugins import silero

import agent as A

MIC_RATE = 16000
FRAME = MIC_RATE // 100  # 10 ms
VOICE = "am_michael"  # a different Kokoro voice from the agent's, as the "user"
AUDIO_DIR = A.ROOT / "data" / "bench_audio"
OUT = A.ROOT / "data" / "bench.jsonl"
ONLY: set[str] = set()
API = os.getenv("MEMORY_API_URL", "http://127.0.0.1:8000")

SEED = [
    "User's dog is called Biscuit.",
    "User is flying to Lisbon next month.",
    "User drinks a matcha latte every morning.",
    "User's sister is called Maya.",
    "User is allergic to peanuts.",
    "User is applying for AI engineer jobs in Toronto.",
    "User goes to a climbing gym on weekends.",  # reworded after run bench-20261114-1620 (never asked about)
    "User is learning French.",
    "User's favourite band is Radiohead.",
    "User's birthday is in March.",
]

# (id, category, text, expected word in a correct reply). recall = needs memory; chat = doesn't;
# write = new fact to store
SCRIPT = [
    ("r1", "recall", "What's my dog's name?", "biscuit"),
    ("c1", "chat", "Tell me a fun fact about octopuses.", None),
    ("r2", "recall", "Where am I flying next month?", "lisbon"),
    ("c2", "chat", "How long should I boil an egg?", None),
    ("r3", "recall", "What do I drink in the morning?", "matcha"),
    ("w1", "write", "Please remember that my favourite colour is green.", None),
    ("r4", "recall", "What's my sister called?", "maya"),
    ("c3", "chat", "What's the capital of Australia?", None),
    ("r5", "recall", "Is there anything I'm allergic to?", "peanut"),
    ("c4", "chat", "Give me one quick stretching tip.", None),
    ("r6", "recall", "What kind of jobs am I applying for?", "engineer"),
    ("w2", "write", "Remember that my cousin Leo is visiting in December.", None),
    ("r7", "recall", "Which language am I learning?", "french"),
    ("c5", "chat", "How do you say thank you in French?", None),
    ("r8", "recall", "What's my favourite band?", "radiohead"),
    ("c6", "chat", "What's a good way to fall asleep faster?", None),
]


# -- fake audio devices ------------------------------------------------------------------------

class ScriptedMic(io.AudioInput):
    """Real-time 16 kHz mono source: silence, plus whatever say() queued."""

    def __init__(self) -> None:
        super().__init__(label="scripted-mic")
        self._pending = np.zeros(0, dtype=np.int16)
        self._drained = asyncio.Event()
        self._t_next: float | None = None

    async def say(self, pcm16k: np.ndarray) -> None:
        self._drained.clear()
        self._pending = np.concatenate([self._pending, pcm16k])
        await self._drained.wait()

    async def __anext__(self) -> rtc.AudioFrame:
        now = time.perf_counter()
        self._t_next = now if self._t_next is None else self._t_next + FRAME / MIC_RATE
        if self._t_next > now:
            await asyncio.sleep(self._t_next - now)
        chunk, self._pending = self._pending[:FRAME], self._pending[FRAME:]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
            if not self._drained.is_set() and len(self._pending) == 0:
                self._drained.set()
        return rtc.AudioFrame(chunk.tobytes(), MIC_RATE, 1, FRAME)


class RealtimeSink(io.AudioOutput):
    """Discards audio but reports playback as if it played in real time (like a speaker would)."""

    def __init__(self) -> None:
        super().__init__(label="realtime-sink", capabilities=io.AudioOutputCapabilities(pause=False),
                         sample_rate=24000)
        self._dur = 0.0
        self._start = 0.0
        self._task: asyncio.Task | None = None
        self._interrupt = asyncio.Event()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if not self._dur:
            self._start = time.monotonic()
            self.on_playback_started(created_at=time.time())
        self._dur += frame.duration

    def flush(self) -> None:
        super().flush()
        if self._dur:
            self._interrupt.clear()
            self._task = asyncio.create_task(self._playout(self._dur))

    async def _playout(self, dur: float) -> None:
        try:
            await asyncio.wait_for(self._interrupt.wait(), max(0.0, self._start + dur - time.monotonic()))
            pos, interrupted = min(time.monotonic() - self._start, dur), True
        except asyncio.TimeoutError:
            pos, interrupted = dur, False
        self._dur = 0.0
        self.on_playback_finished(playback_position=pos, interrupted=interrupted)

    def clear_buffer(self) -> None:
        if self._dur:
            self._interrupt.set()


class LocalInference:
    """In-process stand-in for the worker's inference process, which normally runs the
    turn-detector model. Same runner classes, called in a thread instead of over IPC."""

    def __init__(self) -> None:
        self._runners: dict[str, _InferenceRunner] = {}

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        if method not in self._runners:
            runner = _InferenceRunner.registered_runners[method]()
            await asyncio.to_thread(runner.initialize)
            self._runners[method] = runner
        return await asyncio.to_thread(self._runners[method].run, data)


# -- setup helpers -----------------------------------------------------------------------------

def _post(path: str, body: dict, token: str | None = None) -> dict:
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{API}{path}", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"{}")


def fresh_seeded_user(seed: list[str] = SEED) -> tuple[str, str]:
    """New throwaway account on the LOCAL memory API, seeded with SEED (same facts for every mode)."""
    email, pw = f"bench-{secrets.token_hex(4)}@example.test", secrets.token_urlsafe(18)
    _post("/auth/register", {"email": email, "password": pw})
    token = _post("/auth/login", {"email": email, "password": pw})["access_token"]
    for fact in seed:
        _post("/me/memories", {"text": fact, "importance": 0.7}, token)
    return email, pw


def prompt_audio(pid: str, text: str) -> np.ndarray:
    """Synthesize each question once with Kokoro (cached), return 16 kHz int16."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    f = AUDIO_DIR / f"{pid}.pcm"
    if not f.exists():
        req = urllib.request.Request(f"{A.SPEECH_URL}/audio/speech",
                                     json.dumps({"input": text, "voice": VOICE, "response_format": "pcm"}).encode(),
                                     {"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            f.write_bytes(r.read())
    a24 = np.frombuffer(f.read_bytes(), dtype="<i2").astype(np.float32)
    n = int(len(a24) * MIC_RATE / 24000)
    return np.interp(np.linspace(0, len(a24), n, endpoint=False), np.arange(len(a24)), a24).astype(np.int16)


async def wait_reply_done(session: AgentSession, timeout: float = 40) -> None:
    """Wait for the agent to start and then finish speaking."""
    spoke = asyncio.Event()
    done = asyncio.Event()

    def on_state(ev) -> None:
        if ev.new_state == "speaking":
            spoke.set()
        elif spoke.is_set() and ev.new_state == "listening":
            done.set()

    session.on("agent_state_changed", on_state)
    try:
        await asyncio.wait_for(done.wait(), timeout)
    finally:
        session.off("agent_state_changed", on_state)


async def run_pass(mode: str, user: tuple[str, str], p: int, vad, run_id: str) -> None:
    """One pass of the script in a brand-new session (empty chat history), so an answer can only
    come from memory, never from an earlier pass still sitting in the context."""
    os.environ["MEMORY_EMAIL"], os.environ["MEMORY_PASSWORD"] = user
    session = A.build_session(vad)
    agent = A.MemoryAgent(mode)
    state = A.turn_logger(session, agent, run=run_id, path=OUT)
    mic, sink = ScriptedMic(), RealtimeSink()
    session.input.audio = mic
    session.output.audio = sink
    await session.start(agent=agent, record=False)
    await asyncio.sleep(2)  # let VAD/turn detector settle on silence
    try:
        for pid, cat, text, expect in SCRIPT:
            if ONLY and pid not in ONLY:
                continue
            state["meta"] = {"prompt_id": pid, "category": cat, "prompt": text, "expect": expect, "pass": p}
            waiter = asyncio.create_task(wait_reply_done(session))
            await mic.say(prompt_audio(pid, text))
            try:
                await waiter
            except asyncio.TimeoutError:
                print(f"  [{mode}] {pid}: no reply within 40 s", flush=True)
            await asyncio.sleep(1.0)  # pause before the next question
        print(f"  [{mode}] pass {p} done", flush=True)
    finally:
        await session.aclose()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["none", "tool", "prefetch"])
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="prompt ids to run, e.g. r1 c1 (default: all)")
    args = ap.parse_args()
    global ONLY
    ONLY = set(args.only or [])
    for pid, _, text, _ in SCRIPT:  # synthesize all prompts up front, outside any timing
        prompt_audio(pid, text)
    vad = silero.VAD.load()
    users = {m: fresh_seeded_user() for m in args.modes}  # one seeded account per mode
    run_id = f"bench-{time.strftime('%Y%m%d-%H%M')}"
    with fake_job_context(inference_executor=LocalInference()):
        for p in range(args.passes):
            # rotate the order every pass so no mode always runs first (cold caches) or last
            order = args.modes[p % len(args.modes):] + args.modes[:p % len(args.modes)]
            for mode in order:
                await run_pass(mode, users[mode], p, vad, run_id)
    print(f"wrote {OUT} (run {run_id})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())

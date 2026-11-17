"""Simulated phone calls: the real phone code path, without a phone line.

Runs the same pieces a LiveKit SIP call uses -- caller_account (one memory account per number),
MemoryAgent in PHONE_MEMORY_MODE with PHONE_RULES, turn_logger and call_recorder -- but the
"caller" is a synthesized voice pushed through a telephone-band filter (8 kHz, 300-3400 Hz)
into the live STT -> LLM -> TTS pipeline. No LiveKit Cloud, no number, no cost.

Script: caller A tells the agent two facts and hangs up; caller A calls back and asks for them;
caller B (another number) asks the same questions and must NOT get A's answers.

Run:  uv run phone_sim.py        (needs the memory API, speech server and Ollama, as start.ps1)
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import time

import numpy as np
from livekit.agents.testing import fake_job_context
from livekit.plugins import silero

import agent as A
import bench as B

VOICE_A, VOICE_B = "am_michael", "af_bella"
# made-up 555 numbers, fresh per run so both callers start with empty memory
RUN = secrets.token_hex(2)
CALLER_A, CALLER_B = f"+1416555{RUN}1", f"+1647555{RUN}2"

CALLS = [
    ("A", CALLER_A, VOICE_A, "call1", [
        ("a1", "Hi. Please remember that my sister's name is Lena.", None),
        ("a2", "And remember that I'm allergic to shellfish.", None),
        ("a3", "Okay, thanks. Bye.", None),
    ]),
    ("A", CALLER_A, VOICE_A, "call2", [
        ("b1", "Hi, it's me again. What's my sister's name?", "lena"),
        ("b2", "What am I allergic to?", "shellfish"),
        ("b3", "Great, bye.", None),
    ]),
    ("B", CALLER_B, VOICE_B, "call3", [
        ("c1", "Hi. What's my sister's name?", "!lena"),
        ("c2", "What am I allergic to?", "!shellfish"),
        ("c3", "Okay, bye.", None),
    ]),
]


def phone_band(pcm16k: np.ndarray) -> np.ndarray:
    """What a phone line does to a voice: down to 8 kHz and band-limit to 300-3400 Hz, then back
    up to the 16 kHz the pipeline expects (the extra band stays empty, as on a real call)."""
    x = pcm16k.astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / B.MIC_RATE)
    spec[(freqs < 300) | (freqs > 3400)] = 0
    y = np.fft.irfft(spec, len(x))
    y8 = y[::2]  # 8 kHz
    y16 = np.interp(np.arange(len(y8) * 2) / 2, np.arange(len(y8)), y8)
    return np.clip(y16, -32768, 32767).astype(np.int16)


class FakeCtx:
    """The one JobContext method call_recorder needs."""

    def __init__(self) -> None:
        self.callbacks = []

    def add_shutdown_callback(self, cb) -> None:
        self.callbacks.append(cb)


async def run_call(who: str, number: str, voice: str, label: str, lines, vad) -> dict:
    B.VOICE = voice
    creds = await asyncio.to_thread(A.caller_account, number)
    session = A.build_session(vad)
    agent = A.MemoryAgent(A.PHONE_MEMORY_MODE, creds, A.PHONE_RULES)
    cid = A.caller_id(number)
    started = time.time()
    call = {"id": time.strftime("%Y%m%d-%H%M%S") + f"-{cid}-sim",
            "caller": cid, "kind": "phone-sim", "started": started}
    state = A.turn_logger(session, agent, run=call["id"])
    ctx = FakeCtx()
    A.call_recorder(ctx, session, agent, call)
    mic, sink = B.ScriptedMic(), B.RealtimeSink()
    session.input.audio, session.output.audio = mic, sink
    await session.start(agent=agent, record=False)
    greet = asyncio.create_task(B.wait_reply_done(session))
    await session.generate_reply(instructions=(
        "Answer the phone in one short sentence: say you're the memory assistant and ask "
        "how you can help."))
    await greet
    await asyncio.sleep(0.5)
    results = []
    try:
        for pid, text, expect in lines:
            state["meta"] = {"caller": cid, "kind": "phone-sim", "prompt_id": pid, "expect": expect}
            n_before = len(session.history.items)
            waiter = asyncio.create_task(B.wait_reply_done(session))
            await mic.say(phone_band(B.prompt_audio(f"phone-{voice}-{pid}", text)))
            try:
                await waiter
            except asyncio.TimeoutError:
                pass
            new = [m for m in session.history.items[n_before:]
                   if getattr(m, "type", None) == "message" and m.role == "assistant"]
            reply = " ".join(m.text_content or "" for m in new)
            heard = [m.text_content for m in session.history.items[n_before:]
                     if getattr(m, "type", None) == "message" and m.role == "user"]
            ok = None
            if expect:
                want = expect.lstrip("!")
                ok = (want in reply.lower()) != expect.startswith("!")
            results.append({"pid": pid, "said": text, "heard": " ".join(heard), "reply": reply,
                            "expect": expect, "ok": ok})
            print(f"  [{label} {who}] {pid}  heard: {' '.join(heard)!r}\n"
                  f"{'':14}reply: {reply!r}" + ("" if ok is None else f"  -> {'PASS' if ok else 'FAIL'}"),
                  flush=True)
            await asyncio.sleep(1.0)
    finally:
        await session.aclose()
        for cb in ctx.callbacks:  # what LiveKit runs at hang-up
            await cb()
    return {"call": call["id"], "who": who, "results": results}


async def main() -> None:
    vad = silero.VAD.load()
    for who, number, voice, label, lines in CALLS:  # synthesize up front, outside the timing
        for pid, text, _ in lines:
            B.VOICE = voice
            B.prompt_audio(f"phone-{voice}-{pid}", text)
    out = []
    with fake_job_context(inference_executor=B.LocalInference()):
        for who, number, voice, label, lines in CALLS:
            print(f"--- {label}: caller {who} ({A.caller_id(number)})", flush=True)
            out.append(await run_call(who, number, voice, label, lines, vad))
            await asyncio.sleep(2)
    checks = [r for c in out for r in c["results"] if r["ok"] is not None]
    print(f"\nchecks passed: {sum(r['ok'] for r in checks)}/{len(checks)}")
    (A.ROOT / "data" / "phone_sim_last.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())

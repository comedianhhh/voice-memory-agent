"""Turn the recorded demo (video/demo/) into video/demo.mp4: animation + the real audio of both sides.

1. builds a video-time schedule from timeline.json (session gap compressed into a title card,
   everything inside a session at real speed) -> demo/schedule.js for demo-animation.html
2. renders the page frame by frame (export.render_mp4)
3. mixes user_*.wav / agent_*.wav at their scheduled times and muxes them in

Run:  uv run --no-project --with playwright --with imageio-ffmpeg video/export_demo.py
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from export import render_mp4  # noqa: E402

DEMO = HERE / "demo"
INTRO, PRE, AFTER, SWITCH, OUTRO = 3.5, 0.8, 1.2, 2.8, 2.5


def fact_from_hits(hits: str | None) -> str | None:
    m = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', hits or "")
    return json.loads(f'"{m.group(1)}"') if m else None


def build_schedule() -> dict:
    tl = json.loads((DEMO / "timeline.json").read_text(encoding="utf-8"))
    turns = [e for e in tl["events"] if e["type"] == "turn"]
    out, clips = [], []
    v_cursor = INTRO
    switch = None
    for i, tn in enumerate(turns):
        if i > 0:  # a new session: title card in between
            switch = [v_cursor, v_cursor + SWITCH]
            v_cursor += SWITCH
        base = tn["user_t"] - PRE
        v = lambda r, b=base, c=v_cursor: c + (r - b)  # noqa: E731
        agent_v = v(tn["agent_t"])
        stop_v = agent_v - tn["e2e"]  # when VAD heard the user stop (the Kokoro clip has trailing silence)
        end_v = v(tn["agent_t"] + tn["agent_dur"]) + AFTER

        parts = [f'memory search · {tn["prefetch_ms"]:.0f} ms → ']
        hit = fact_from_hits(tn["memory_hits"])
        parts[0] += f"“{html.escape(hit)}”" if hit else "nothing stored yet"
        for name, args in zip(tn["tools"], tn["tool_args"]):
            text = json.loads(args).get("text", "")
            parts.append(f"<b>{name}</b> → “{html.escape(text)}”")
        chip = "<br>".join(f"<span>{p}</span>" for p in parts)

        label = ("Session 1 <span>memory is empty</span>" if tn["session"] == 1
                 else f'Session {tn["session"]} <span>new session · empty chat history</span>')
        out.append({
            "heard": tn["heard"], "reply": tn["reply"], "chip_html": chip, "session_label": label,
            "session_start": v_cursor - 0.4, "session_end": end_v,
            "user_v": v(tn["user_t"]), "stop_v": stop_v, "user_end_v": stop_v,
            "chip_v": stop_v + 0.55 * (agent_v - stop_v),
            "agent_v": agent_v, "agent_end_v": v(tn["agent_t"] + tn["agent_dur"]),
            "e2e": tn["e2e"],
        })
        clips += [(DEMO / f'user_{tn["id"]}.wav', v(tn["user_t"])), (DEMO / f'agent_{tn["id"]}.wav', agent_v)]
        v_cursor = end_v
    out[-1]["session_end"] += OUTRO
    sched = {"duration": round(v_cursor + OUTRO, 2), "intro_end": INTRO, "switch": switch, "turns": out}
    (DEMO / "schedule.js").write_text("window.DEMO = " + json.dumps(sched, indent=1) + ";\n", encoding="utf-8")
    return {"sched": sched, "clips": clips}


def main() -> None:
    s = build_schedule()
    silent = HERE / "demo-silent.mp4"
    render_mp4((HERE / "demo-animation.html").resolve().as_uri(), silent)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    args = [ff, "-y", "-loglevel", "error", "-i", str(silent)]
    filters, labels = [], []
    for k, (wav, at) in enumerate(s["clips"]):
        args += ["-i", str(wav)]
        ms = int(at * 1000)
        filters.append(f"[{k + 1}:a]aresample=48000,adelay={ms}|{ms}[a{k}]")
        labels.append(f"[a{k}]")
    filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,apad[aout]")
    out = HERE / "demo.mp4"
    args += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(s["sched"]["duration"]), str(out)]
    subprocess.run(args, check=True)
    silent.unlink()
    print(f"wrote {out} ({s['sched']['duration']} s)")


if __name__ == "__main__":
    main()

"""Turn data/bench.jsonl (or any turns log) into p50 / p95 tables per memory mode.

Run:  uv run stats.py [path] [--run RUN_ID]      -> prints Markdown, also writes data/results.md

Latency is in ms. p95 on ~16-32 turns per cell is the 2nd-worst value or so: read it as
"the slow tail", not a precise percentile.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGES = [
    ("end_of_turn", "end of turn"),
    ("transcription", "STT final"),
    ("prefetch", "memory prefetch"),
    ("llm_ttft", "LLM first token"),
    ("tts_ttfb", "TTS first audio"),
    ("e2e", "**end-to-end**"),
]
MODES = ["none", "tool", "prefetch"]


def pct(xs: list[float], q: float) -> str:
    return f"{np.percentile(xs, q) * 1000:.0f}" if xs else "–"


def load(path: Path, run: str | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if run:
        rows = [r for r in rows if r.get("run") == run]
    for r in rows:
        exp = r.get("expect")
        r["correct"] = None if not exp else exp in (r.get("reply") or "").lower()
    return rows


def table(rows: list[dict], title: str) -> str:
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    modes = [m for m in MODES if m in by_mode]
    out = [f"### {title}", "", "| stage (ms) | " + " | ".join(f"{m} p50 | {m} p95" for m in modes) + " |",
           "|---|" + "---|---|" * len(modes)]
    for key, label in STAGES:
        cells = []
        for m in modes:
            xs = [r[key] for r in by_mode[m] if r.get(key) is not None]
            cells += [pct(xs, 50), pct(xs, 95)]
        if any(c != "–" for c in cells):
            out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("| turns | " + " | ".join(f"{len(by_mode[m])} | " for m in modes) + "|")
    return "\n".join(out)


def accuracy(rows: list[dict]) -> str:
    out = ["### Recall accuracy (reply contains the seeded fact)", "",
           "| mode | correct | tool calls per recall turn |", "|---|---|---|"]
    for m in MODES:
        rec = [r for r in rows if r["mode"] == m and r.get("category") == "recall"]
        if not rec:
            continue
        ok = sum(1 for r in rec if r["correct"])
        calls = np.mean([len(r.get("tools") or []) for r in rec])
        out.append(f"| {m} | {ok}/{len(rec)} ({ok / len(rec):.0%}) | {calls:.2f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/bench.jsonl")
    ap.add_argument("--run")
    a = ap.parse_args()
    rows = load(Path(a.path), a.run)
    recall = [r for r in rows if r.get("category") == "recall"]
    other = [r for r in rows if r.get("category") in ("chat", "write")]
    md = "\n\n".join([
        table(rows, "All turns"),
        table(recall, "Recall questions only (\"what's my dog's name?\")"),
        table(other, "Turns that need no lookup (chit-chat and \"remember that…\")"),
        accuracy(rows),
    ])
    print(md)
    Path("data/results.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

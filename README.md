# voice-memory-agent

A real-time voice assistant that remembers you across conversations — running entirely on one
desktop GPU, no paid APIs — plus a benchmark that measures **what long-term memory costs a voice
turn in milliseconds**, and how reliably it gets the answer right.

**[▶ 85-second demo video](video/voice-memory-agent-demo.mp4)**: a real two-session run (tell it
two facts → close → new session asks for them), then the benchmark results.

```
 mic ─▶ VAD + turn detector ─▶ Whisper STT ─▶ LLM (Ollama) ─▶ Kokoro TTS ─▶ speaker
                                               │  ▲
                                   MCP (stdio) ▼  │
                                  memory-service (Postgres + pgvector, RLS, hybrid search)
```

| piece | what | where |
|---|---|---|
| orchestration | [LiveKit Agents](https://github.com/livekit/agents) 1.8 cascade pipeline | `agent.py` |
| STT / TTS | faster-whisper large-v3-turbo + Kokoro-82M on the GPU, behind an OpenAI-compatible API I wrote | `servers/speech_server.py` |
| LLM | qwen3:8b on Ollama, thinking off | |
| memory | my [memory-service](https://github.com/comedianhhh/memory-service), exposed as an MCP server | `MemoryAgent` in `agent.py` |
| benchmark | scripted spoken questions through the real pipeline, no mic | `bench.py`, `stats.py` |

## Three ways to give a voice agent memory

`MEMORY_MODE` switches between them; everything else stays identical.

- **tool**: the LLM decides when to call `search_memory` / `write_memory`. A lookup costs an extra
  LLM round trip, and only happens if the model remembers to do it.
- **prefetch**: every user turn runs a memory search *before* the LLM, and the hits go into the
  prompt. No extra LLM round, but every turn pays for a search.
- **none**: no memory (baseline).

## Results

Run `bench-20261114-1620`: 3 modes × 3 passes × 16 spoken questions = 144 turns. Each pass is a
fresh session, so answers can't come from chat history. Mode order rotates every pass. Local
RTX 4080 SUPER, qwen3:8b, Whisper large-v3-turbo, Kokoro. Times in ms, measured from when the
user stops speaking.

**Recall questions** ("What's my dog's name?"), 24 per mode:

| | none | tool | prefetch |
|---|---|---|---|
| correct answers | 0 / 24 | 22 / 24 | **24 / 24** |
| end-to-end p50 | 1175 | 1500 (**+325**) | 1214 (**+39**) |
| end-to-end p95 | 1293 | 1744 | 2013 |

**Turns that need no memory** (chit-chat, "remember that…"), 24 per mode:

| | none | tool | prefetch |
|---|---|---|---|
| end-to-end p50 | 1141 | 1200 (+59) | 1289 (**+148**) |
| memory writes made when asked | – | 6 / 6 | 4 / 6 |

**Where the time goes** (all 48 turns per mode, p50):

| stage | none | tool | prefetch |
|---|---|---|---|
| end of turn (silence → "user is done") | 935 | 947 | 914 |
| memory search before the LLM | – | – | 47 |
| LLM first token | 32 | 49 | 172 |
| TTS first audio | 118 | 56 | 57 |
| **end-to-end** | **1148** | **1434** | **1264** |

What it says:

1. **Prefetch wins on recall, by a lot.** Answers come back 286 ms sooner than with a tool call
   (one LLM round instead of two), and accuracy is 24/24 vs 22/24. Both tool-mode misses were
   "What's my favourite band?". The model searched, but with a query and `k` it picked itself,
   and the right fact didn't come back. Prefetch always uses the user's own words and k=3.
2. **But prefetch taxes every turn.** On turns that need no memory it still searches, adding
   ~148 ms p50, and its p95 is the worst of the three. The tail is the first search of each session
   (MCP server spawn, login, embedding model load: 560–820 ms). Warming that up at session start
   should remove most of it. It also wrote 2 fewer memories: with no search tool in view, the model
   reached for `write_memory` less often.
3. **Memory isn't the bottleneck. Turn-taking is.** ~930 ms of the ~1150–1450 ms end-to-end goes
   to deciding the user has finished speaking: VAD silence, then the non-streaming Whisper
   transcript, then the turn-detector model. LLM and TTS together are ~100–200 ms. The next
   optimisation is endpointing, not the model.

Raw per-turn data: [`results/bench-20261114-1620.jsonl`](results/bench-20261114-1620.jsonl)
(`uv run stats.py results/bench-20261114-1620.jsonl` reproduces the tables). The first, flawed run
is kept too ([`results/bench-v1-flawed-shared-session.jsonl`](results/bench-v1-flawed-shared-session.jsonl)).
It reused one session across passes, so the second pass could answer from chat history instead of
memory: tool mode searched on only half its recall turns. Fixed by giving every pass a fresh
session and rotating the mode order.

Caveats: a single machine, synthetic voice for the user, 24 turns per cell (p95 ≈ the 2nd-worst
turn). TTS first-audio time depends on the length of the reply's first sentence, so it varies
with what the model says.

## Phone calls

The same agent can answer a phone number through [LiveKit Cloud](https://livekit.com/pricing)
telephony. The free Build plan includes one US local number and 50 inbound minutes, with no card
on file. Only the call audio goes through LiveKit: STT, LLM, TTS and memory still run on this
machine, and the worker connects out to LiveKit, so no ports are opened.

- **Memory follows the caller.** Each caller number gets its own memory-service account, created on
  the first call (`caller_account` in `agent.py`), so RLS keeps one caller's memories away from
  another's. Logs carry a SHA-256 hash of the number, never the number itself.
- **Every call is logged.** At hang-up, `data/calls/<id>.json` gets the duration and the full
  transcript. Per-turn latency for the same call goes to `data/turns.jsonl` under the call id.
- **Phone settings.** Calls use prefetch mode (the benchmark winner), plus a phone rule: audio is
  narrowband, so ask the caller to repeat what wasn't caught.

**Simulated test** (`phone_sim.py`, no phone line). This runs the real phone code path. The caller
is a synthesized voice squeezed to phone quality (8 kHz, 300–3400 Hz) and played into the live
pipeline. Caller A states two facts and hangs up, then calls back and asks for them. Caller B, on
another number, asks the same questions.

| call | asked | agent said | check |
|---|---|---|---|
| A, call back | "What's my sister's name?" | "Your sister's name is Lena." | ✅ |
| A, call back | "What am I allergic to?" | "You're allergic to shellfish." | ✅ |
| B, new number | "What's my sister's name?" | "I don't remember yet. …" | ✅ isolated |
| B, new number | "What am I allergic to?" | "I don't remember yet. …" | ✅ isolated |

Whisper transcribed every phone-band line correctly. End-to-end p50 was 1,145 ms over 9 turns, in
line with the benchmark. The slowest turn (2,978 ms) was the first one, where end-of-turn
detection alone took 2.5 s: endpointing again, not the model. Raw records:
[`results/phone-sim-20261116.jsonl`](results/phone-sim-20261116.jsonl). Not yet tested over a real
LiveKit number: the SIP leg, the dispatch rule, and reading `sip.phoneNumber` in `entrypoint`.

To go live: create a LiveKit Cloud project and put its `LIVEKIT_*` keys in `.env` (see
`.env.example`). Then claim the free number and add a dispatch rule that sends calls to the agent
named `voice-memory-agent`. Start everything with `.\start.ps1 -Phone`.

## How the benchmark works

`bench.py` doesn't use a microphone. A fake mic (`ScriptedMic`) plays pre-synthesized questions
into the session in real time, as 10 ms frames with silence in between. That means VAD,
end-of-turn detection, STT, LLM, memory and TTS all run exactly as in a live call. A fake speaker
(`RealtimeSink`) "plays" each reply in real time, so turn-taking stays realistic. Every mode gets
a fresh local account seeded with the same 10 facts, then the same 16 questions: 8 recall,
6 chit-chat, 2 "remember that…". Per-stage latency comes from LiveKit's own `ChatMessage.metrics`
and is logged per turn to `data/*.jsonl`.

**end-to-end** = from when the user stops speaking to when agent audio starts.

## Run it

```powershell
.\start.ps1                       # Docker (memory DB+API), speech server, then talk in the console
uv run memory_test.py tool        # text-only check: remember in one session, recall in another
uv run bench.py --passes 2        # benchmark all modes -> data/bench.jsonl
uv run stats.py                   # p50/p95 tables -> data/results.md
uv run phone_sim.py               # 3 simulated phone calls: recall on call-back + caller isolation
.\start.ps1 -Phone                # answer real calls to your LiveKit number (LIVEKIT_* in .env)
```

The demo video is generated, not screen-recorded. `video/demo_record.py` runs the real
two-session conversation with a synthesized user and records both voices, the tool calls and the
latencies. `video/export_demo.py` animates that recording and mixes in the audio, and
`video/export.py` renders the results animation. Both renderers drive a headless Chrome frame by
frame.

Needs: an NVIDIA GPU (tested on a 16 GB RTX 4080 SUPER), Ollama, Docker Desktop, `uv`, and
memory-service checked out next to this folder.

## Things I found along the way

- **`localhost` cost 2.3 s per LLM call on Windows.** It resolves to IPv6 first and Ollama only
  listens on IPv4. Switching to `127.0.0.1` took LLM time-to-first-token from ~2.4 s to ~0.05 s.
- **qwen3 must have thinking off** (`reasoning_effort="none"`), or the first token takes 2–8 s.
- **A small model under-calls tools.** With a generic "call search_memory when the user asks about
  themselves", qwen3:8b answered "I don't remember" to *What's my dog's name?* without searching.
  It needed an explicit "you know nothing about the user except what search_memory returns" rule.
- **The first turn of a fresh session took 7.5 s.** Ollama unloads the model after 5 idle minutes,
  and the memory MCP server only spawned and logged in on the first search (~1 s). `MemoryAgent.warm_up`
  now does both when the session starts, and pins the model for 30 min. Same turn afterwards: 2.5 s.
  It's a write turn, so two LLM rounds. The memory lookup went from 1082 ms to 48 ms. (The benchmark
  above ran before this fix, so its prefetch p95 still includes that cold start.)
- **memory-service ranks by relevance only.** Two contradicting facts ("dog is Noodle" /
  "dog is Waffles") came back oldest-first. That needs recency weighting or an update path.

_AI-assisted: I set the design, the experiments and the acceptance checks; Claude wrote most of
the code._

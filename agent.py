"""Voice agent with long-term memory: LiveKit Agents cascade (STT -> LLM -> TTS), all local by default.

  STT  faster-whisper   via servers/speech_server.py  (OpenAI-compatible, port 8880)
  LLM  Ollama qwen3:8b  via its OpenAI-compatible /v1 (thinking off)
  TTS  Kokoro           via servers/speech_server.py; TTS_PROVIDER=elevenlabs swaps in ElevenLabs
  Memory  memory-service's MCP server (stdio), three ways to use it -- MEMORY_MODE:
    tool      the LLM calls search_memory / write_memory itself when it decides to (default)
    prefetch  every user turn searches memory first and puts the hits in the prompt;
              the LLM only gets write_memory, so reads never cost a second LLM round
    none      no memory at all (baseline)

Run (talk through your mic/speakers, no LiveKit server needed):
  uv run agent.py download-files     # once: VAD + turn-detector weights
  uv run agent.py console            # or .\\start.ps1, which also starts the services

Every turn's latency goes to data/turns.jsonl (see turn_logger).

Always use 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and each
request to an IPv4-only server (Ollama) loses ~2 s. Measured 2026-11-14: LLM time-to-first-token
2.4 s with localhost, ~0.1 s with 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    function_tool,
    llm,
    mcp,
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.english import EnglishModel
from mcp.client.stdio import get_default_environment

load_dotenv()
log = logging.getLogger("voice-memory-agent")

ROOT = Path(__file__).parent
SPEECH_URL = os.getenv("SPEECH_URL", "http://127.0.0.1:8880/v1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "kokoro")  # kokoro | elevenlabs
MEMORY_MODE = os.getenv("MEMORY_MODE", "tool")  # tool | prefetch | none
MEMORY_SERVICE_DIR = os.getenv("MEMORY_SERVICE_DIR", str(ROOT.parent / "memory-service"))
TURNS_LOG = Path(os.getenv("TURNS_LOG", ROOT / "data" / "turns.jsonl"))
PREFETCH_K = 3

BASE = """You are a friendly voice assistant with long-term memory. Your replies are spoken aloud, so:
- keep them to one or two short sentences,
- no markdown, lists, emoji or symbols that cannot be read out.
"""
WRITE_RULE = """- When the user tells you something about themselves worth keeping (preferences, plans, people,
  facts about their life) or says "remember", call write_memory with one short third-person fact,
  e.g. "User's favourite drink is a matcha latte."
"""
INSTRUCTIONS = {
    "tool": BASE + "\nMemory tools:\n" + WRITE_RULE + """- You know nothing about the user except what search_memory returns. For ANY question about the
  user (their name, pets, family, friends, plans, travel, habits, food, health, work, hobbies,
  preferences, or anything they told you before), call search_memory FIRST, then answer from the
  results. Never say you don't remember without searching first.
""",
    "prefetch": BASE + "\nMemory:\n" + WRITE_RULE + """- With each user message you get a system note with memories that may be relevant. When the
  user asks about themselves, answer from those notes; if none fits, say you don't remember yet.
""",
    "none": BASE,
}


def memory_server() -> mcp.MCPServerStdio:
    """memory-service's MCP server over stdio, run with memory-service's own venv.

    It calls the memory-service HTTP API as a normal logged-in user, so RLS keeps each
    user's memories separate. Credentials come from .env (a local test account).
    """
    py = Path(MEMORY_SERVICE_DIR) / ".venv" / "Scripts" / "python.exe"
    env = get_default_environment() | {
        "MEMORY_API_URL": os.getenv("MEMORY_API_URL", "http://127.0.0.1:8000"),
        "MEMORY_EMAIL": os.environ["MEMORY_EMAIL"],
        "MEMORY_PASSWORD": os.environ["MEMORY_PASSWORD"],
    }
    # first search loads the embedding model, so allow more than the 5 s default
    return mcp.MCPServerStdio(str(py), ["-m", "memory_service.mcp_server"], env=env,
                              cwd=MEMORY_SERVICE_DIR, client_session_timeout_seconds=20)


def memory_tools() -> mcp.MCPToolset:
    return mcp.MCPToolset(id="memory", mcp_server=memory_server())


def _ollama_keep_loaded(keep_alive: str = "30m") -> None:
    """Load the model now and keep it in VRAM (Ollama's default unloads it after 5 idle minutes).
    An empty prompt to the native API only loads the model, it generates nothing."""
    base = OLLAMA_URL.removesuffix("/v1")
    body = json.dumps({"model": LLM_MODEL, "keep_alive": keep_alive}).encode()
    req = urllib.request.Request(f"{base}/api/generate", body, {"content-type": "application/json"})
    urllib.request.urlopen(req, timeout=60).close()


def _tool_text(result: Any) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content).strip()


class MemoryAgent(Agent):
    """One agent class for all three memory modes, so the benchmark compares only the memory path."""

    def __init__(self, mode: str = MEMORY_MODE) -> None:
        self.mode = mode
        self._server: mcp.MCPServerStdio | None = None
        tools: list = []
        if mode == "tool":
            tools = [memory_tools()]
        elif mode == "prefetch":
            self._server = memory_server()
            tools = [self.write_memory_tool()]
        self.last_prefetch_ms: float | None = None
        super().__init__(instructions=INSTRUCTIONS[mode], tools=tools)

    # -- prefetch mode -------------------------------------------------------------------------
    async def _call(self, name: str, args: dict) -> str:
        assert self._server is not None
        if not self._server.initialized:
            await self._server.initialize()
        # MCPServer has no public single-call API; _call_tool is what MCPToolset uses inside
        return _tool_text(await self._server._call_tool(name, args))

    def write_memory_tool(self):
        agent = self

        @function_tool
        async def write_memory(text: str, importance: float = 0.5) -> str:
            """Store one short fact about the user for later (max 4000 chars). importance 0..1."""
            return await agent._call("write_memory", {"text": text, "importance": importance})

        return write_memory

    async def llm_node(self, chat_ctx: llm.ChatContext, tools, model_settings):
        """Prefetch lives here, not in on_user_turn_completed: that hook only fires for spoken turns,
        while llm_node runs for every reply (voice and text tests alike). It searches only when the
        newest item is the user's message, i.e. not on the follow-up call after a write_memory."""
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if (self.mode == "prefetch" and getattr(last, "type", None) == "message"
                and last.role == "user" and last.text_content):
            t0 = time.perf_counter()
            hits = await self._call("search_memory", {"query": last.text_content, "k": PREFETCH_K})
            self.last_prefetch_ms = (time.perf_counter() - t0) * 1000
            self.last_prefetch_hits = hits
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(role="system",
                                 content=f"Possibly relevant memories about the user:\n{hits or '(none)'}")
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def on_enter(self) -> None:
        # warm-up in the background, so the first question doesn't pay for cold starts
        self._warmup_task = asyncio.create_task(self.warm_up())

    async def warm_up(self) -> None:
        """Pay the cold starts before the user's first question, not during it. Measured
        2026-11-14: the first turn of a fresh session took 7.5 s end to end because (1) Ollama
        had unloaded the model after 5 idle minutes and (2) the memory MCP server only spawned
        and logged in on the first search (~1 s)."""
        t0 = time.perf_counter()
        await asyncio.to_thread(_ollama_keep_loaded)
        if self.mode == "prefetch":
            await self._call("search_memory", {"query": "warm-up", "k": 1})
        elif self.mode == "tool":
            for ts in self.tools:
                if isinstance(ts, mcp.MCPToolset):
                    await ts.setup()
                    srv = ts._mcp_server
                    await srv._call_tool("search_memory", {"query": "warm-up", "k": 1})  # forces login
        log.info("warm-up done in %.0f ms", (time.perf_counter() - t0) * 1000)

    async def on_exit(self) -> None:
        if self._server is not None:
            await self._server.aclose()


def turn_logger(session: AgentSession, agent: MemoryAgent, *, run: str = "live", path: Path = TURNS_LOG):
    """Append one JSON line per assistant reply with the per-stage latency LiveKit measured.

    Stages (seconds, from ChatMessage.metrics): end_of_turn (VAD silence -> turn decided),
    transcription (speech end -> final transcript), llm_ttft, tts_ttfb, e2e (user stopped
    speaking -> agent audio started). Tool calls in between are counted, not timed separately:
    with a tool call, llm_ttft is the *second* LLM call and the tool round trip only shows in e2e.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"user": None, "tools": []}

    @session.on("function_tools_executed")
    def _tools(ev) -> None:
        state["tools"] += [c.name for c in ev.function_calls]

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        item = ev.item
        if getattr(item, "type", "message") != "message":
            return
        if item.role == "user":
            state["user"] = item
            state["tools"] = []
            return
        if item.role != "assistant" or state["user"] is None:
            return  # e.g. the greeting
        u, a = state["user"].metrics or {}, item.metrics or {}
        row = {
            "run": run,
            "mode": agent.mode,
            "llm": LLM_MODEL,
            "tts": TTS_PROVIDER,
            "user": state["user"].text_content,
            "reply": item.text_content,
            "tools": state["tools"],
            "end_of_turn": u.get("end_of_turn_delay"),
            "transcription": u.get("transcription_delay"),
            "prefetch": (agent.last_prefetch_ms or 0) / 1000 if agent.mode == "prefetch" else None,
            "llm_ttft": a.get("llm_node_ttft"),
            "tts_ttfb": a.get("tts_node_ttfb"),
            "e2e": a.get("e2e_latency"),
            "interrupted": bool(item.interrupted),
            "ts": time.time(),
            **(state.get("meta") or {}),  # bench.py sets the scripted prompt's id/category here
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        state["user"] = None

    return state


def build_tts():
    if TTS_PROVIDER == "elevenlabs":
        from livekit.plugins import elevenlabs  # needs ELEVEN_API_KEY in .env

        return elevenlabs.TTS(model="eleven_flash_v2_5")
    return openai.TTS(base_url=SPEECH_URL, api_key="local", model="kokoro",
                      voice="af_heart", response_format="pcm")


def build_session(vad) -> AgentSession:
    return AgentSession(
        vad=vad,
        stt=openai.STT(base_url=SPEECH_URL, api_key="local", model="whisper-1", language="en"),
        llm=openai.LLM(base_url=OLLAMA_URL, api_key="ollama", model=LLM_MODEL,
                       reasoning_effort="none"),
        tts=build_tts(),
        turn_handling={"turn_detection": EnglishModel()},
    )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = build_session(ctx.proc.userdata["vad"])
    agent = MemoryAgent()
    turn_logger(session, agent)
    await session.start(agent=agent, room=ctx.room, record=False)
    await session.generate_reply(instructions="Greet the user in one short sentence.")


if __name__ == "__main__":
    cli.run_app(server)

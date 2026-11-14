"""Cross-session memory test, text only (no mic): tell session 1 a fact, ask session 2 for it.

Needs: Ollama, memory-service API (docker compose up -d db api), .env with the test account.
Run: uv run memory_test.py [tool|prefetch]   (default: tool)
"""

from __future__ import annotations

import asyncio
import os
import json
import secrets
import sys
import urllib.request

from livekit.agents import AgentSession

import agent as A

# a random detail, so a pass can't come from an older run's memory or the model guessing
CITY = secrets.choice(["Lisbon", "Kyoto", "Oaxaca", "Tallinn", "Hobart", "Valparaiso"])
DOG = secrets.choice(["Biscuit", "Pickle", "Mochi", "Waffles", "Noodle"])


def fresh_user() -> None:
    """Register a throwaway account on the LOCAL memory API so every run starts with no memories
    (the API has no delete, and old runs' facts would otherwise compete in search)."""
    email = f"voice-test-{secrets.token_hex(4)}@example.test"
    password = secrets.token_urlsafe(18)
    url = os.getenv("MEMORY_API_URL", "http://127.0.0.1:8000")
    body = json.dumps({"email": email, "password": password}).encode()
    req = urllib.request.Request(f"{url}/auth/register", body, {"content-type": "application/json"})
    urllib.request.urlopen(req).close()  # raises on non-2xx
    os.environ["MEMORY_EMAIL"], os.environ["MEMORY_PASSWORD"] = email, password


def new_session() -> AgentSession:
    llm = A.openai.LLM(base_url=A.OLLAMA_URL, api_key="ollama", model=A.LLM_MODEL,
                       reasoning_effort="none")
    return AgentSession(llm=llm)


async def turn(session: AgentSession, text: str) -> tuple[list[str], str]:
    result = await session.run(user_input=text)
    tools, reply = [], ""
    for ev in result.events:
        item = getattr(ev, "item", None)
        if ev.type == "function_call":
            tools.append(f"{item.name}{item.arguments}")
        elif ev.type == "message" and item.role == "assistant":
            reply = item.text_content or ""
    print(f"  user : {text}\n  tools: {tools}\n  agent: {reply}\n")
    return tools, reply


async def main(mode: str) -> None:
    fresh_user()
    print(f"memory mode: {mode}")
    print("session 1")
    async with new_session() as s1:
        await s1.start(A.MemoryAgent(mode))
        tools, _ = await turn(s1, f"Please remember that my dog is called {DOG} and I'm flying to {CITY} next month.")
        assert any(t.startswith("write_memory") for t in tools), "agent did not write a memory"

    print("session 2 (fresh, empty chat history)")
    async with new_session() as s2:
        await s2.start(A.MemoryAgent(mode))
        tools, reply = await turn(s2, "Where am I travelling next month, and what's my dog's name?")
        if mode == "tool":  # in prefetch mode the search happens before the LLM, not as a tool call
            assert any(t.startswith("search_memory") for t in tools), "agent did not search memory"
        assert CITY.lower() in reply.lower() and DOG.lower() in reply.lower(), reply
    print(f"PASS: recalled {DOG} and {CITY} across sessions")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # model replies can contain emoji; cp1252 can't print them
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "tool"))

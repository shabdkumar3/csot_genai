"""
Research Desk - Week 3 Project
===============================
Class hierarchy:
  Agent       - brain: chat(), _run_loop(), dispatch(), session I/O. no UI.
  REPLAgent   - terminal REPL + one-shot CLI
  TUIAgent    - Textual UI (in tui.py, not imported here at module load time)

Usage:
  python agent.py                              # REPLAgent.run()
  python agent.py "What is quantum computing?" # REPLAgent.run_once()
  python agent.py --tui                        # TUIAgent.run()
  python agent.py --session abc123 "continue"
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

from tools import files as files_tools
from tools import papers as papers_tools
from tools import web as web_tools

load_dotenv()

# --- config ---

SESSIONS_DIR = os.path.join(".agent", "sessions")
AGENTS_PATHS = ("AGENTS.md", os.path.join(".agent", "AGENTS.md"))
BASE_PROMPT = (
    "You are Research Desk, a research assistant. Answer using tools, not from memory - "
    "search before you answer factual or current questions. If a tool errors, try a "
    "different angle instead of giving up or apologising."
)

MAX_ITERATIONS = 10
MAX_TOOL_RESULT_CHARS = 16_000  # safety net on top of each tool's own truncation

MODEL = "llama-3.3-70b-versatile"
client = OpenAI(
    api_key=os.getenv("groq_api_key"),
    base_url="https://api.groq.com/openai/v1",
)

# merge all three tool modules into one registry the Agent dispatches against
TOOLS = files_tools.TOOLS + papers_tools.TOOLS + web_tools.TOOLS
DISPATCH = {**files_tools.DISPATCH, **papers_tools.DISPATCH, **web_tools.DISPATCH}


# --- sessions (episodic memory) ---

def create_session() -> str:
    """new 8-char session id. directory is created lazily on first save."""
    return uuid.uuid4().hex[:8]


def save_session(session_id: str, messages: list, title: str = "Untitled") -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    created_at = datetime.now(timezone.utc).isoformat()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            created_at = json.load(f).get("created_at", created_at)

    data = {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_session(session_id: str) -> dict | None:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    out = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
            d = json.load(f)
        out.append({"id": d["id"], "title": d.get("title", "Untitled"), "updated_at": d.get("updated_at", "")})
    return sorted(out, key=lambda s: s["updated_at"], reverse=True)


# --- procedural memory ---

def build_system_prompt() -> str:
    """base prompt + AGENTS.md if it exists - same pattern opencode/claude code use."""
    parts = [BASE_PROMPT]
    for path in AGENTS_PATHS:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f"## Project rules\n{f.read()}")
            break
    return "\n\n".join(parts)


def _short(args) -> str:
    s = json.dumps(args) if not isinstance(args, str) else args
    return s if len(s) < 80 else s[:80] + "..."


# --- the brain ---

class Agent:
    """Loop, tools, sessions. No input()/print()/Textual in here."""

    def __init__(self, workspace: str = ".", session_id: str | None = None):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(os.path.join(self.workspace, "notes"), exist_ok=True)

        if session_id:
            session = load_session(session_id)
            if session is None:
                raise ValueError(f"no such session: {session_id}")
            self.session_id = session_id
            self.title = session.get("title", "Untitled")
            self.messages = session["messages"]
        else:
            self.session_id = create_session()
            self.title = "Untitled"
            self.messages = [{"role": "system", "content": build_system_prompt()}]

    def chat(self, user_message: str) -> str:
        """append user turn, run the tool loop, save, return the final answer."""
        is_first_exchange = len(self.messages) == 1  # only the system prompt so far
        self.messages.append({"role": "user", "content": user_message})

        answer = self._run_loop()

        if is_first_exchange:
            self._autotitle(user_message)

        save_session(self.session_id, self.messages, self.title)
        return answer

    def run_once(self, prompt: str) -> str:
        return self.chat(prompt)

    def _run_loop(self) -> str:
        for _ in range(MAX_ITERATIONS):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS,
                temperature=0.3,
            )
            msg = resp.choices[0].message

            entry = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(entry)

            if not msg.tool_calls:
                return msg.content or "(no answer)"

            for tc in msg.tool_calls:
                self._emit("tool_call", name=tc.function.name, arguments=tc.function.arguments)
                result = self.dispatch(tc)
                self._emit("tool_result", name=tc.function.name, result=result)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "stopped: hit the iteration limit before finishing - try narrowing the question"

    def dispatch(self, tool_call) -> str:
        """route a tool call to the right python function, return a json string."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "tool call had invalid arguments json"})

        fn = DISPATCH.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool: {name}"})

        try:
            result = fn(**args)
        except Exception as e:
            result = {"error": f"{name} failed: {e}"}

        out = json.dumps(result)
        return out[:MAX_TOOL_RESULT_CHARS]

    def _autotitle(self, first_message: str) -> None:
        """bonus: ask the model for a short title after the first exchange."""
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Give a plain 5-word-max title (no quotes, no punctuation) for a research session that starts with:\n\n{first_message}",
                }],
                temperature=0,
                max_tokens=20,
            )
            title = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
            if title:
                self.title = title
        except Exception:
            pass  # "Untitled" is a fine fallback, not worth crashing the chat over

    def _emit(self, event: str, **data) -> None:
        """hook for subclasses: REPLAgent prints to stderr, TUIAgent writes to a log panel."""
        pass


class REPLAgent(Agent):
    """terminal REPL + one-shot CLI. /sessions and /resume <id> are the bonus commands."""

    def run(self) -> None:
        print(f"Research Desk [session {self.session_id}: {self.title}]")
        print("/quit to exit · /sessions to list · /resume <id> to switch\n")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                break
            if user_input == "/sessions":
                self._print_sessions()
                continue
            if user_input.startswith("/resume"):
                self._resume(user_input[len("/resume"):].strip())
                continue

            print(self.chat(user_input))
            print()

    def _print_sessions(self) -> None:
        sessions = list_sessions()
        if not sessions:
            print("  (no sessions yet)")
            return
        for s in sessions:
            marker = "*" if s["id"] == self.session_id else " "
            print(f" {marker} {s['id']}  {s['title']}  ({s['updated_at']})")

    def _resume(self, session_id: str) -> None:
        if not session_id:
            print("usage: /resume <session_id>")
            return
        session = load_session(session_id)
        if session is None:
            print(f"no such session: {session_id}")
            return
        self.session_id = session_id
        self.title = session.get("title", "Untitled")
        self.messages = session["messages"]
        print(f"resumed [{session_id}]: {self.title}")

    def _emit(self, event: str, **data) -> None:
        if event == "tool_call":
            print(f"  [tool] {data.get('name')} {_short(data.get('arguments'))}", file=sys.stderr)
        elif event == "tool_result":
            print(f"  [tool] -> {len(str(data.get('result')))} chars", file=sys.stderr)


def main():
    args = sys.argv[1:]
    session_id = None
    tui = False
    prompt_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--session" and i + 1 < len(args):
            session_id = args[i + 1]
            i += 2
        elif args[i] == "--tui":
            tui = True
            i += 1
        else:
            prompt_parts.append(args[i])
            i += 1

    if tui:
        # imported lazily so plain REPL/one-shot use never pulls in textual
        from tui import TUIAgent
        TUIAgent(session_id=session_id).run()
        return

    agent = REPLAgent(session_id=session_id)
    if prompt_parts:
        print(agent.run_once(" ".join(prompt_parts)))
        return
    agent.run()


if __name__ == "__main__":
    main()

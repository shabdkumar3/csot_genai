# research agent - week 2 (everything in one file)
import os
import asyncio
from datetime import datetime

import requests
import trafilatura
from markdownify import markdownify as md
from openai import AsyncOpenAI, BadRequestError
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, RichLog

load_dotenv()


MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 8
UA = "Mozilla/5.0 (research-agent)"
MCP_URL = "https://api.alphaxiv.org/mcp/v1"

SYSTEM_PROMPT = """You are a research assistant. You answer using tools, not from memory.

Tools you have:
- web_search(query): find web results (titles, links, snippets). Use this first.
- web_fetch(url): read the full cleaned text of a page.
- AlphaXiv paper tools (discover_papers, get_paper_content, ...): for academic / arXiv questions.

How to work:
1. General question -> web_search, then web_fetch the best 1-2 links.
2. "Papers on X" / academic question -> discover_papers, then get_paper_content on the most relevant ones.
3. Keep calling tools until you actually have enough to answer. Do not guess.
4. Then write a clear, structured answer and list the sources (URLs / arXiv IDs) you used.

Be factual and concise. If a tool errors, try a different angle instead of giving up."""

_client = None


def _ensure_client():
    # groq, same as week 1
    global _client
    if _client is None:
        key = os.getenv("groq_api_key")
        if not key:
            raise RuntimeError("no key found - set groq_api_key in .env")
        _client = AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return _client, MODEL


def web_search(query, num_results=5):
    # serper search
    key = os.environ.get("serper_api_key")
    if not key:
        return "web_search error: serper_api_key not set"

    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return f"web_search error: {e}"

    out = []

    # quick answer if present
    box = data.get("answerBox")
    if box:
        ans = box.get("answer") or box.get("snippet")
        if ans:
            out.append(f"[answer] {ans}")

    # organic results
    for item in data.get("organic", [])[:num_results]:
        out.append(f"- {item.get('title')}\n  {item.get('link')}\n  {item.get('snippet', '')}")

    return "\n".join(out) if out else "no results found"


def web_fetch(url, max_chars=6000):
    # read a page
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        return f"web_fetch error: {e}"

    html = r.text

    # clean main text
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        # fallback
        text = md(html, strip=["script", "style"])

    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"

    return text or "(empty page)"


WEB_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (Serper/Google). Returns titles, links and snippets. Use this first to find sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to search for"},
                    "num_results": {"type": "integer", "description": "how many results, default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its cleaned text. Use after web_search to read a page in full.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "the URL to read"}},
                "required": ["url"],
            },
        },
    },
]

LOCAL_TOOLS = {"web_search": web_search, "web_fetch": web_fetch}


def alphaxiv_params():
    # spawn the bridge (handles oauth)
    return StdioServerParameters(command="npx", args=["-y", "mcp-remote", MCP_URL])


def to_openai_tools(listed):
    # mcp schema -> openai schema
    tools = []
    for t in listed.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


async def call_mcp(session, name, args):
    # run a paper tool
    result = await session.call_tool(name, args)
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else "(no content returned)"


def _short(args):
    import json
    s = json.dumps(args)
    return s if len(s) < 80 else s[:80] + "..."


async def run(query, history, emit=None):
    emit = emit or (lambda line: None)
    history.append({"role": "user", "content": query})

    # only the connect part is in here, so a later loop error isnt blamed on mcp
    connected = False
    try:
        async with stdio_client(alphaxiv_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                mcp_names = {t.name for t in listed.tools}
                tools = WEB_TOOL_SCHEMAS + to_openai_tools(listed)
                emit(f"[green]alphaxiv connected[/] ({len(mcp_names)} paper tools)")
                connected = True
                return await _loop(history, tools, session, mcp_names, emit)
    except Exception as e:
        if connected:
            raise  # came from the loop, not the connection
        emit(f"[yellow]alphaxiv unavailable[/] - running web only ({type(e).__name__})")

    # web-only (reached only if the connection failed)
    return await _loop(history, WEB_TOOL_SCHEMAS, None, set(), emit)


async def _loop(history, tools, session, mcp_names, emit):
    import json
    client, model = _ensure_client()

    for _ in range(MAX_STEPS):
        # ask the model. llama sometimes sends a weird tool format so just retry
        resp = None
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=history, tools=tools, temperature=0.3
                )
                break
            except BadRequestError as e:
                if "tool_use_failed" in str(e) and attempt < 2:
                    emit("[dim]  (bad tool format, retrying...)[/]")
                    continue
                raise
        msg = resp.choices[0].message

        # save assistant turn
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
        history.append(entry)

        # no tools -> final answer
        if not msg.tool_calls:
            return msg.content or "(no answer)"

        # run each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            emit(f"[cyan]{name}[/] {_short(args)}")

            try:
                if name in LOCAL_TOOLS:
                    # web tools block, push to a thread
                    result = await asyncio.to_thread(LOCAL_TOOLS[name], **args)
                elif session is not None and name in mcp_names:
                    result = await call_mcp(session, name, args)
                else:
                    result = f"unknown tool: {name}"
            except Exception as e:
                result = f"{name} error: {e}"

            emit(f"[dim]  -> {len(str(result))} chars[/]")
            history.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "stopped: hit step limit before finishing"


class ResearchApp(App):
    CSS = """
    Horizontal {
        height: 1fr;
    }
    #chat {
        width: 65%;
        border: round $primary;
        padding: 0 1;
    }
    #tools {
        width: 35%;
        border: round $warning;
        padding: 0 1;
    }
    Input {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear view"),
        Binding("ctrl+k", "clear_history", "New chat"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield RichLog(id="chat", wrap=True, markup=True)
            yield RichLog(id="tools", wrap=True, markup=True)
        yield Input(placeholder="ask a research question...")
        yield Footer()

    def on_mount(self):
        chat = self.query_one("#chat", RichLog)
        tools = self.query_one("#tools", RichLog)
        chat.border_title = "research"
        tools.border_title = "tool activity"
        chat.write("[bold]ResearchBot[/bold] - your own little perplexity")
        chat.write("[dim]ctrl+l clear view · ctrl+k new chat · ctrl+s save · ctrl+q quit[/dim]\n")
        self.query_one(Input).focus()

    # helpers (ui thread)
    def _write_chat(self, text):
        self.query_one("#chat", RichLog).write(text)

    def _write_tool(self, text):
        self.query_one("#tools", RichLog).write(text)

    def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            return

        if self._busy:
            self._write_chat("[yellow]still working on the last one...[/yellow]")
            return

        event.input.clear()
        self._write_chat(f"\n[bold cyan]you[/bold cyan]  {text}")
        self._write_tool(f"[dim]--- {text[:40]} ---[/dim]")
        self._busy = True
        self.run_query(text)

    @work(thread=True)
    def run_query(self, text):
        # runs the agent off the ui thread
        def emit(line):
            self.call_from_thread(self._write_tool, line)

        try:
            answer = asyncio.run(run(text, self.history, emit))
            self.call_from_thread(self._write_chat, f"[bold green]agent[/bold green]  {answer}")
        except Exception as e:
            self.call_from_thread(self._write_chat, f"[red]error[/red] {e}")
        finally:
            self._busy = False

    # key actions
    def action_clear_display(self):
        # wipe view, keep memory
        self.query_one("#chat", RichLog).clear()
        self.query_one("#tools", RichLog).clear()
        self._write_chat("[dim]view cleared (history kept)[/dim]")

    def action_clear_history(self):
        # full reset
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.query_one("#chat", RichLog).clear()
        self.query_one("#tools", RichLog).clear()
        self._write_chat("[dim]new chat - history cleared[/dim]")

    def action_save(self):
        # dump transcript
        name = f"transcript_{datetime.now():%Y%m%d_%H%M%S}.md"
        with open(name, "w", encoding="utf-8") as f:
            for m in self.history:
                role = m.get("role")
                if role in ("user", "assistant") and m.get("content"):
                    f.write(f"**{role}:** {m['content']}\n\n")
        self.notify(f"saved to {name}")


if __name__ == "__main__":
    ResearchApp().run()
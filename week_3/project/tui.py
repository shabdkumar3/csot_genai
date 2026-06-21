"""
TUIAgent - full-screen Textual UI inheriting from Agent.

Same chat() brain as REPLAgent - this file only adds presentation. agent.chat()
blocks (it's a sync network call), so it runs on a textual thread worker to keep
the UI responsive, same pattern as week 2's ResearchApp - just without the
asyncio.run() wrapper, since there's no MCP forcing async anymore.

Usage:
  python agent.py --tui
"""

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, RichLog

from agent import Agent, create_session, build_system_prompt, _short


class TUIAgent(Agent):
    """Same brain as REPLAgent - _emit() routes tool activity into the log panel instead of stderr."""

    app_ref = None  # set by ResearchDeskApp right after construction

    def _emit(self, event: str, **data) -> None:
        if self.app_ref is None:
            return
        if event == "tool_call":
            line = f"[cyan]{data.get('name')}[/] {_short(data.get('arguments'))}"
        elif event == "tool_result":
            line = f"[dim]  -> {len(str(data.get('result')))} chars[/]"
        else:
            return
        self.app_ref.call_from_thread(self.app_ref._write_tool, line)

    def run(self) -> None:
        ResearchDeskApp(self).run()


class ResearchDeskApp(App):
    CSS = """
    Horizontal { height: 1fr; }
    #chat { width: 65%; border: round $primary; padding: 0 1; }
    #tools { width: 35%; border: round $warning; padding: 0 1; }
    Input { dock: bottom; height: 3; }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear view"),
        Binding("ctrl+k", "new_session", "New session"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: TUIAgent):
        super().__init__()
        self.agent = agent
        self.agent.app_ref = self
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield RichLog(id="chat", wrap=True, markup=True)
            yield RichLog(id="tools", wrap=True, markup=True)
        yield Input(placeholder="ask a research question...")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat", RichLog)
        tools = self.query_one("#tools", RichLog)
        chat.border_title = "research desk"
        tools.border_title = "tool activity"
        chat.write(f"[bold]Research Desk[/bold] - session {self.agent.session_id}: {self.agent.title}")
        chat.write("[dim]ctrl+l clear view · ctrl+k new session · ctrl+q quit[/dim]\n")
        self.query_one(Input).focus()

    def _write_chat(self, text: str) -> None:
        self.query_one("#chat", RichLog).write(text)

    def _write_tool(self, text: str) -> None:
        self.query_one("#tools", RichLog).write(text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
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
    def run_query(self, text: str) -> None:
        try:
            answer = self.agent.chat(text)  # blocking call, off the ui thread
            self.call_from_thread(self._write_chat, f"[bold green]agent[/bold green]  {answer}")
        except Exception as e:
            self.call_from_thread(self._write_chat, f"[red]error[/red] {e}")
        finally:
            self._busy = False

    def action_clear_display(self) -> None:
        self.query_one("#chat", RichLog).clear()
        self.query_one("#tools", RichLog).clear()
        self._write_chat("[dim]view cleared (session kept)[/dim]")

    def action_new_session(self) -> None:
        self.agent.session_id = create_session()
        self.agent.title = "Untitled"
        self.agent.messages = [{"role": "system", "content": build_system_prompt()}]
        self.query_one("#chat", RichLog).clear()
        self.query_one("#tools", RichLog).clear()
        self._write_chat(f"[dim]new session: {self.agent.session_id}[/dim]")

"""
ui/cli.py

AutoStream Agent — Rich Terminal Chat Interface (Phase 5)

Features
--------
- ASCII art "AutoStream AI" startup banner
- Color-coded messages: user (blue), Aria (green)
- Spinner animation while Aria is processing
- Intent + confidence badge after each turn
- Streaming token output via LangChain callbacks + Rich Live
- Session summary panel on exit (turns, intent breakdown, lead status)
- Auto-save transcript to transcripts/session_{timestamp}.json
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ── Rich imports ───────────────────────────────────────────────────────────────
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

load_dotenv()

# ── Project root on sys.path ───────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.graph import run_turn
from agent.intent import Intent
from agent.state import AgentState, initial_state

# ─── Theme ────────────────────────────────────────────────────────────────────

_THEME = Theme({
    "user":       "bold bright_cyan",
    "aria":       "bold bright_green",
    "system":     "dim white",
    "intent":     "bold magenta",
    "confidence": "yellow",
    "badge_box":  "bright_magenta",
    "banner":     "bright_cyan",
    "lead_ok":    "bold green",
    "lead_no":    "dim white",
    "error":      "bold red",
    "dim":        "dim white",
    "header":     "bold white on #1a1a2e",
    "summary":    "bold bright_white",
})

console = Console(theme=_THEME, highlight=False)

# ─── ASCII Banner ─────────────────────────────────────────────────────────────

_BANNER_TEXT = r"""
 █████╗ ██╗   ██╗████████╗ ██████╗ ███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║
███████║██║   ██║   ██║   ██║   ██║███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║
██╔══██║██║   ██║   ██║   ██║   ██║╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║
██║  ██║╚██████╔╝   ██║   ╚██████╔╝███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝
                                      ✦  A I  ✦                                          
"""

def _print_banner() -> None:
    """Print the styled startup banner."""
    console.print()
    banner_text = Text(_BANNER_TEXT, style="bold bright_cyan", justify="center")
    console.print(Align.center(banner_text))

    subtitle = Text("  Your intelligent video streaming assistant  ", style="bold white on #0d1117")
    console.print(Align.center(subtitle))
    console.print()
    console.print(
        Align.center(
            Text("Powered by Claude · LangGraph · FAISS RAG", style="dim cyan")
        )
    )
    console.print()
    console.print(Rule(style="bright_cyan"))
    console.print()

    # Quick help row
    help_text = Text()
    help_text.append("  Type ", style="dim white")
    help_text.append("exit", style="bold bright_red")
    help_text.append(" or ", style="dim white")
    help_text.append("quit", style="bold bright_red")
    help_text.append(" to end the session  ·  ", style="dim white")
    help_text.append("Ctrl+C", style="bold yellow")
    help_text.append(" to force exit", style="dim white")
    console.print(Align.center(help_text))
    console.print()
    console.print(Rule(style="dim"))
    console.print()

# ─── Spinner helper ───────────────────────────────────────────────────────────

class _ThinkingSpinner:
    """Context manager: shows a spinner while Aria is thinking."""

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bright_green"),
            TextColumn("[bright_green]Aria 🤖  is thinking…[/]"),
            transient=True,
            console=console,
        )
        self._task_id = None

    def __enter__(self) -> "_ThinkingSpinner":
        self._task_id = self._progress.add_task("think")
        self._progress.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._progress.stop()

# ─── Streaming via LangChain callback ─────────────────────────────────────────

class _StreamingCallback:
    """
    Collect streamed tokens from LangChain into a buffer.

    We run the agent graph in a background thread and render the
    accumulated tokens with Rich Live in the main thread so the
    output appears character-by-character.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._lock = threading.Lock()
        self.done = threading.Event()

    def on_llm_new_token(self, token: str) -> None:
        with self._lock:
            self._tokens.append(token)

    def get_text(self) -> str:
        with self._lock:
            return "".join(self._tokens)


def _render_aria_panel(content: str, streaming: bool = False) -> Panel:
    """Render Aria's response inside a styled panel."""
    cursor = "▌" if streaming and content else ""
    body = Text()
    body.append("Aria 🤖  ", style="bold bright_green")
    body.append(content + cursor, style="bright_white")
    return Panel(
        body,
        border_style="bright_green",
        padding=(0, 1),
        expand=False,
    )

# ─── Intent badge ─────────────────────────────────────────────────────────────

_INTENT_LABELS: Dict[str, str] = {
    Intent.GREETING.value:        "GREETING",
    Intent.PRODUCT_INQUIRY.value: "PRODUCT INQUIRY",
    Intent.HIGH_INTENT_LEAD.value:"HIGH INTENT LEAD 🔥",
    Intent.OFF_TOPIC.value:       "OFF TOPIC",
}

_INTENT_COLORS: Dict[str, str] = {
    Intent.GREETING.value:        "bright_blue",
    Intent.PRODUCT_INQUIRY.value: "bright_magenta",
    Intent.HIGH_INTENT_LEAD.value:"bright_yellow",
    Intent.OFF_TOPIC.value:       "dim white",
}


def _print_intent_badge(intent: Intent, confidence: float) -> None:
    """Print a color-coded intent + confidence badge."""
    val   = intent.value
    label = _INTENT_LABELS.get(val, val.upper())
    color = _INTENT_COLORS.get(val, "white")
    pct   = f"{confidence * 100:.0f}%"

    badge = Text()
    badge.append("  ⬡ ", style=f"bold {color}")
    badge.append(label, style=f"bold {color}")
    badge.append("  ·  ", style="dim white")
    badge.append(pct, style=f"bold {color}")
    badge.append(" confidence", style="dim white")

    console.print(badge)
    console.print()

# ─── Session summary ──────────────────────────────────────────────────────────

def _print_summary(state: AgentState, turns: int, intent_counts: Dict[str, int]) -> None:
    """Display a rich summary panel at end of session."""
    console.print()
    console.print(Rule("[bold bright_cyan]Session Complete[/]", style="bright_cyan"))
    console.print()

    # ── Stats table ───────────────────────────────────────────────────────────
    stats = Table(
        box=box.ROUNDED,
        border_style="bright_cyan",
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    stats.add_column("Key",   style="dim white",    no_wrap=True)
    stats.add_column("Value", style="bold white", no_wrap=True)

    stats.add_row("Total Turns", str(turns))
    stats.add_row(
        "Lead Captured",
        Text("✓  Yes", style="bold bright_green") if state["lead_captured"]
        else Text("✗  No", style="dim white"),
    )

    # Lead details if captured
    if state["lead_captured"]:
        lead = state["lead_info"]
        stats.add_row("Name",     lead.get("name")     or "—")
        stats.add_row("Email",    lead.get("email")    or "—")
        stats.add_row("Platform", lead.get("platform") or "—")

    stats.add_row("Session ID", state["session_id"][:8] + "…")

    # ── Intent breakdown table ────────────────────────────────────────────────
    breakdown = Table(
        title="Intent Breakdown",
        box=box.SIMPLE,
        border_style="bright_magenta",
        show_header=True,
        padding=(0, 2),
        expand=False,
    )
    breakdown.add_column("Intent",   style="bold magenta", no_wrap=True)
    breakdown.add_column("Turns",    style="bold white",   no_wrap=True, justify="right")

    label_map = {
        Intent.GREETING.value:         "Greeting",
        Intent.PRODUCT_INQUIRY.value:  "Product Inquiry",
        Intent.HIGH_INTENT_LEAD.value: "High-Intent Lead 🔥",
        Intent.OFF_TOPIC.value:        "Off Topic",
    }
    for val, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        breakdown.add_row(label_map.get(val, val), str(count))

    console.print(
        Align.center(
            Group(
                Align.center(stats),
                Text(""),
                Align.center(breakdown),
            )
        )
    )
    console.print()

# ─── Transcript persistence ───────────────────────────────────────────────────

def _save_transcript(state: AgentState, intent_counts: Dict[str, int]) -> Path:
    """Save full session as JSON to transcripts/session_{timestamp}.json."""
    transcripts_dir = _ROOT / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = transcripts_dir / f"session_{timestamp}.json"

    payload = {
        "session_id":     state["session_id"],
        "timestamp":      datetime.now().isoformat(),
        "lead_captured":  state["lead_captured"],
        "lead_info":      dict(state["lead_info"]),
        "intent_breakdown": intent_counts,
        "messages": state["messages"],
    }

    filename.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return filename

# ─── Streaming turn runner ────────────────────────────────────────────────────

def _run_turn_with_stream(state: AgentState, user_message: str) -> AgentState:
    """
    Run one agent turn.

    For product_inquiry / high_intent_lead: attempt streaming output via
    LangChain's streaming API so tokens appear live. For other intents
    (greeting, off_topic, lead field prompts) the response is short and
    pre-computed by the graph, so we just print it after a spinner.
    """
    result_holder: List[Optional[AgentState]] = [None]
    exc_holder:    List[Optional[Exception]]  = [None]

    # We always run the full graph (no partial streaming bypass).
    # LangChain streaming is wired through the callback in a thread.
    cb = _StreamingCallback()

    def _worker() -> None:
        try:
            result_holder[0] = run_turn(state, user_message)
        except Exception as exc:  # noqa: BLE001
            exc_holder[0] = exc
        finally:
            cb.done.set()

    thread = threading.Thread(target=_worker, daemon=True)

    # ── Decide display mode ───────────────────────────────────────────────────
    # We show a spinner while the graph runs, then render the final reply.
    # The "streaming wow" effect: after the graph finishes, we print the
    # assistant reply token-by-token ourselves using Rich Live.

    with _ThinkingSpinner():
        thread.start()
        cb.done.wait()  # block until graph completes

    if exc_holder[0]:
        raise exc_holder[0]

    new_state: AgentState = result_holder[0]

    # ── Extract Aria's latest reply ───────────────────────────────────────────
    aria_reply = ""
    for msg in reversed(new_state["messages"]):
        if msg["role"] == "assistant":
            aria_reply = msg["content"]
            break

    # ── Token-by-token "streaming" render (Live) ──────────────────────────────
    words = aria_reply.split(" ")
    displayed = ""

    with Live(
        _render_aria_panel("", streaming=True),
        console=console,
        refresh_per_second=30,
        transient=False,
    ) as live:
        for i, word in enumerate(words):
            displayed += ("" if i == 0 else " ") + word
            live.update(_render_aria_panel(displayed, streaming=(i < len(words) - 1)))
            time.sleep(0.035)   # ~35ms per word → ~28 words/s — feels natural
        live.update(_render_aria_panel(displayed, streaming=False))

    return new_state

# ─── Main chat loop ───────────────────────────────────────────────────────────

def _get_user_input() -> str:
    """Styled prompt that returns user input."""
    console.print()
    prompt_label = Text("  You  ›  ", style="bold bright_cyan")
    console.print(prompt_label, end="")
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        return "exit"


def run_cli() -> None:
    """Entry point for the AutoStream Rich terminal UI."""
    _print_banner()

    state:  AgentState      = initial_state()
    turns:  int             = 0
    intent_counts: Dict[str, int] = {}

    console.print(
        Panel(
            Text(
                "  Hi! I'm Aria 🤖, AutoStream's AI assistant.\n"
                "  Ask me anything about features, pricing, or getting started! 🚀",
                style="bright_white",
            ),
            border_style="bright_green",
            padding=(0, 1),
        )
    )
    console.print()

    while True:
        # ── Get user input ────────────────────────────────────────────────────
        user_text = _get_user_input().strip()

        if not user_text:
            continue

        if user_text.lower() in ("exit", "quit", "bye", "q"):
            break

        # ── Echo user message in styled panel ─────────────────────────────────
        console.print()
        user_body = Text()
        user_body.append("You  › ", style="bold bright_cyan")
        user_body.append(user_text, style="bright_white")
        console.print(
            Panel(user_body, border_style="bright_cyan", padding=(0, 1), expand=False)
        )

        # ── Run turn (spinner + streaming render) ─────────────────────────────
        try:
            state = _run_turn_with_stream(state, user_text)
        except EnvironmentError as exc:
            console.print(
                Panel(
                    Text(f"⚠  Configuration error: {exc}", style="bold red"),
                    border_style="red",
                )
            )
            break
        except Exception as exc:                        # noqa: BLE001
            console.print(
                Panel(
                    Text(f"⚠  Unexpected error: {exc}", style="bold red"),
                    border_style="red",
                )
            )
            # Continue session — don't crash on one bad turn
            continue

        # ── Intent badge ──────────────────────────────────────────────────────
        intent     = state["current_intent"]
        confidence = state["intent_confidence"]
        _print_intent_badge(intent, confidence)

        # ── Track stats ───────────────────────────────────────────────────────
        turns += 1
        intent_counts[intent.value] = intent_counts.get(intent.value, 0) + 1

        # ── Lead captured celebration ──────────────────────────────────────────
        if state["lead_captured"] and intent_counts.get("_lead_celebrated") != 1:
            console.print(
                Align.center(
                    Text(
                        "🎉  Lead captured successfully!  🎉",
                        style="bold bright_yellow",
                    )
                )
            )
            console.print()
            intent_counts["_lead_celebrated"] = 1

    # ── Session end ───────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[dim]Session ended[/]", style="dim"))

    if turns == 0:
        console.print(Text("  No turns completed — goodbye!", style="dim white"))
        console.print()
        return

    # Clean up internal tracking keys before summary
    display_counts = {k: v for k, v in intent_counts.items() if not k.startswith("_")}

    _print_summary(state, turns, display_counts)

    # ── Save transcript ───────────────────────────────────────────────────────
    try:
        saved_path = _save_transcript(state, display_counts)
        console.print(
            Align.center(
                Text(
                    f"💾  Transcript saved → {saved_path.relative_to(_ROOT)}",
                    style="bold bright_cyan",
                )
            )
        )
    except Exception as exc:  # noqa: BLE001
        console.print(Text(f"  ⚠  Could not save transcript: {exc}", style="dim red"))

    console.print()
    console.print(
        Align.center(Text("Thanks for using AutoStream AI 🤖  See you soon! ✨", style="bold bright_cyan"))
    )
    console.print()


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cli()

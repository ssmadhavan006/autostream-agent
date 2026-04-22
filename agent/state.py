"""
agent/state.py

AgentState is the single source of truth that flows through every
LangGraph node. All fields are defined here and nowhere else.
"""

from __future__ import annotations

import uuid
from typing import Optional
from typing_extensions import TypedDict

from agent.intent import Intent


# ─── Sub-types ────────────────────────────────────────────────────────────────

class LeadInfo(TypedDict):
    """Contact details collected during a HIGH_INTENT_LEAD conversation."""
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]    # e.g. "YouTube", "TikTok", "Instagram"


# ─── AgentState ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Complete state that is threaded through every node in the graph.

    Fields
    ------
    messages        Full conversation history.
                    Each dict: {"role": "user"|"assistant", "content": str}
    current_intent  Most recent classified intent.
    intent_confidence  0.0–1.0 confidence of the last classification.
    lead_info       Collected contact details; values start as None.
    lead_captured   True once mock_lead_capture() has been called successfully.
    rag_context     Concatenated KB chunks retrieved for the last user query.
    awaiting_field  Which lead field the agent is currently asking for.
                    One of: "name" | "email" | "platform" | None
    session_id      UUID string used for transcript export filenames.
    """
    messages: list[dict]
    current_intent: Intent
    intent_confidence: float
    lead_info: LeadInfo
    lead_captured: bool
    rag_context: str
    awaiting_field: Optional[str]
    session_id: str


# ─── Factory ──────────────────────────────────────────────────────────────────

def initial_state(session_id: Optional[str] = None) -> AgentState:
    """Return a clean, default AgentState for a new conversation session."""
    return AgentState(
        messages=[],
        current_intent=Intent.GREETING,
        intent_confidence=1.0,
        lead_info=LeadInfo(name=None, email=None, platform=None),
        lead_captured=False,
        rag_context="",
        awaiting_field=None,
        session_id=session_id or str(uuid.uuid4()),
    )

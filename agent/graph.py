"""
agent/graph.py

AutoStream LangGraph State Machine.

Graph topology
--------------
START
  └─► classify_intent_node
        ├─► [greeting]         ──────────────────────► generate_response_node ──► END
        ├─► [product_inquiry]  ──────────► retrieve_context_node
        │                                       └──► generate_response_node ──► END
        ├─► [high_intent_lead] ──────────► retrieve_context_node
        │                                       └──► collect_lead_node
        │                                               ├─► [fields incomplete] ──► generate_response_node ──► END
        │                                               └─► [all fields filled] ──► capture_lead_node ──► END
        └─► [off_topic]        ──────────────────────► generate_response_node ──► END

Key design choices
------------------
- retrieve_context_node runs for BOTH product_inquiry AND high_intent_lead so
  Aria always has KB context when she prompts for lead fields.
- capture_lead_node directly appends the success message to state["messages"],
  so generate_response_node is skipped after a successful capture (avoids a
  redundant LLM call and double-reply).
- The graph compiles to a reusable callable: graph(state) -> updated_state.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.intent import Intent
from agent.nodes import (
    capture_lead_node,
    classify_intent_node,
    collect_lead_node,
    generate_response_node,
    retrieve_context_node,
)
from agent.state import AgentState


# ─── Router functions ─────────────────────────────────────────────────────────

def intent_router(state: AgentState) -> str:
    """Route after classify_intent_node based on detected intent.

    Returns one of: "need_context", "handle_lead", "generate_response"
    """
    intent = state["current_intent"]
    if intent in (Intent.PRODUCT_INQUIRY, Intent.HIGH_INTENT_LEAD):
        return "need_context"
    # GREETING and OFF_TOPIC go straight to response generation
    return "generate_response"


def post_context_router(state: AgentState) -> str:
    """After retrieve_context_node, decide whether to collect lead or respond."""
    intent = state["current_intent"]
    if intent == Intent.HIGH_INTENT_LEAD and not state.get("lead_captured", False):
        return "collect_lead"
    return "generate_response"


def lead_router(state: AgentState) -> str:
    """After collect_lead_node, decide if we have all fields yet."""
    lead = state["lead_info"]
    all_filled = all(lead.get(f) for f in ("name", "email", "platform"))
    return "capture" if all_filled else "respond"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble and compile the AutoStream agent graph.

    Returns a compiled LangGraph that can be invoked with an AgentState dict.
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("classify_intent",   classify_intent_node)
    workflow.add_node("retrieve_context",  retrieve_context_node)
    workflow.add_node("collect_lead",      collect_lead_node)
    workflow.add_node("capture_lead",      capture_lead_node)
    workflow.add_node("generate_response", generate_response_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.add_edge(START, "classify_intent")

    # ── After classify: route by intent ──────────────────────────────────────
    workflow.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "need_context":      "retrieve_context",
            "generate_response": "generate_response",
        },
    )

    # ── After retrieve_context: route to lead collection or response ──────────
    workflow.add_conditional_edges(
        "retrieve_context",
        post_context_router,
        {
            "collect_lead":      "collect_lead",
            "generate_response": "generate_response",
        },
    )

    # ── After collect_lead: route by completeness ─────────────────────────────
    workflow.add_conditional_edges(
        "collect_lead",
        lead_router,
        {
            "capture": "capture_lead",
            "respond": "generate_response",
        },
    )

    # ── capture_lead writes its own success message; skip generate_response ───
    workflow.add_edge("capture_lead",      END)

    # ── generate_response always ends the turn ────────────────────────────────
    workflow.add_edge("generate_response", END)

    return workflow.compile()


# ─── Module-level singleton ───────────────────────────────────────────────────

graph = build_graph()


# ─── Convenience runner ───────────────────────────────────────────────────────

def run_turn(state: AgentState, user_message: str) -> AgentState:
    """Process one user turn through the graph and return the updated state.

    Args:
        state:        Current AgentState (from previous turn or initial_state()).
        user_message: The raw text the user just typed.

    Returns:
        Updated AgentState after all graph nodes have executed.
    """
    # Append the user message before invoking the graph
    new_messages = list(state["messages"]) + [
        {"role": "user", "content": user_message}
    ]
    input_state = dict(state)
    input_state["messages"] = new_messages

    return graph.invoke(input_state)

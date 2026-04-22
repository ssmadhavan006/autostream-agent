"""
agent/nodes.py

All LangGraph node functions for AutoStream Agent.

Every node signature: (state: AgentState) -> dict
The returned dict is merged into AgentState by LangGraph.

Nodes
-----
classify_intent_node    — Classify the latest user message, update intent fields.
retrieve_context_node   — RAG retrieval; runs only for inquiry / high-intent turns.
generate_response_node  — Main LLM call; produces the reply shown to the user.
collect_lead_node       — Extract & sequence lead fields (name → email → platform).
                          Partial recovery: detects out-of-order fields automatically.
capture_lead_node       — Calls mock_lead_capture when all 3 fields are present.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import logging

from agent.intent import Intent, IntentResult, apply_transition_rules, classify_intent
from agent.state import AgentState, LeadInfo
from agent.tools import mock_lead_capture, validate_email

logger = logging.getLogger(__name__)
from config.settings import (
    ANTHROPIC_API_KEY,
    INTENT_MIN_CONFIDENCE,
    LLM_MODEL,
    RAG_TOP_K,
)
from rag.retriever import NoInfoSignal, get_vectorstore, retrieve

# ─── Shared LLM helper ────────────────────────────────────────────────────────

def _llm(system: str, user: str, max_tokens: int = 512) -> str:
    """Make a synchronous Claude call and return the text content."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.4,
        max_tokens=max_tokens,
    )
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()


# ─── System Prompt for Aria ───────────────────────────────────────────────────

_ARIA_SYSTEM = """\
You are Aria, AutoStream's AI sales assistant.
AutoStream is a cloud-based video streaming platform that helps content creators \
automate video editing, host professional videos, and grow their audience.

Personality: Helpful, warm, and concise. You never use jargon. You sound like a \
knowledgeable human advisor — not a chatbot.

Rules:
  1. Answer ONLY from the Knowledge Base Context provided below.
     Never invent pricing, features, or policies not present in the context.
  2. If the context is empty or insufficient, say:
     "I don't have that information right now. Our team at support@autostream.io can help!"
  3. Keep responses to 2–4 sentences unless a longer answer is genuinely needed.
  4. When the user shows interest in signing up, smoothly transition to collecting
     their name, email, and streaming platform — one field per message.
  5. Always sound excited to help — never robotic.

--- Current Conversation State ---
Intent        : {intent}
Confidence    : {confidence:.0%}
Lead Captured : {lead_captured}
Awaiting Field: {awaiting_field}
Lead Info     : {lead_info}

--- Knowledge Base Context ---
{rag_context}
"""


def _build_aria_system(state: AgentState) -> str:
    lead_info = state["lead_info"]
    return _ARIA_SYSTEM.format(
        intent=state["current_intent"].value,
        confidence=state["intent_confidence"],
        lead_captured=state["lead_captured"],
        awaiting_field=state["awaiting_field"] or "none",
        lead_info=json.dumps(lead_info, indent=None),
        rag_context=state["rag_context"] or "(no context retrieved)",
    )


def _history_to_user_prompt(messages: List[dict], latest: str) -> str:
    """Format conversation history + latest message into a single user prompt."""
    turns = []
    for m in messages[-6:]:          # last 6 messages for context window economy
        role = m["role"].upper()
        turns.append(f"[{role}]: {m['content']}")
    turns.append(f"[USER]: {latest}")
    return "\n".join(turns)


def _latest_user_message(state: AgentState) -> str:
    """Return the most recent user message from the conversation history."""
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# ─── Node 1: classify_intent_node ────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> dict:
    """Classify the latest user message and update intent fields in state.

    Also enforces the sticky HIGH_INTENT_LEAD transition rule by passing
    the previous intent to the classifier.
    """
    message = _latest_user_message(state)
    if not message:
        return {}  # Nothing to classify

    # Build conversation history for context
    history = [(m["role"], m["content"]) for m in state["messages"][:-1]]

    try:
        result: IntentResult = classify_intent(
            message=message,
            history=history,
            previous_intent=state.get("current_intent"),
        )
    except EnvironmentError:
        # No API key — fall back gracefully (useful for offline testing)
        result = IntentResult(
            intent=Intent.PRODUCT_INQUIRY,
            confidence=0.5,
            reasoning="No API key; defaulting to product_inquiry.",
        )

    return {
        "current_intent": result.intent,
        "intent_confidence": result.confidence,
    }


# ─── Node 2: retrieve_context_node ───────────────────────────────────────────

_store = None   # module-level cache so the FAISS index loads only once


def _get_store():
    global _store
    if _store is None:
        _store = get_vectorstore()
    return _store


def retrieve_context_node(state: AgentState) -> dict:
    """Run RAG retrieval for the latest user message and update rag_context.

    Only called when intent is PRODUCT_INQUIRY or HIGH_INTENT_LEAD.
    If no chunks exceed the confidence threshold, sets rag_context to "".
    """
    message = _latest_user_message(state)
    if not message:
        return {"rag_context": ""}

    try:
        store = _get_store()
        results = retrieve(message, k=RAG_TOP_K, store=store)
    except Exception as exc:
        print(f"[nodes] RAG retrieval error: {exc}")
        return {"rag_context": ""}

    if isinstance(results, NoInfoSignal):
        return {"rag_context": ""}

    chunks = "\n\n---\n\n".join(r.document.page_content for r in results)
    return {"rag_context": chunks}


# ─── Node 3: generate_response_node ──────────────────────────────────────────

_LEAD_PROMPT_TEMPLATES = {
    "name":     "Could I get your name to personalise your onboarding? 😊",
    "email":    "Great! And what's the best email address to reach you at?",
    "platform": "Awesome! Which platform do you mainly stream to — YouTube, Instagram, TikTok, or somewhere else?",
}

_OFF_TOPIC_RESPONSE = (
    "Ha, that's a fun one — but I'm specialised in AutoStream questions! "
    "I'd love to tell you about how AutoStream can level up your content. "
    "What would you like to know?"
)


def generate_response_node(state: AgentState) -> dict:
    """Generate Aria's reply to the user using the LLM.

    Branches:
      - OFF_TOPIC      → canned redirect (no LLM call wasted)
      - Awaiting field → prompt for the next lead field
      - Otherwise      → full Aria LLM call with context + history
    """
    messages = state["messages"]
    latest = _latest_user_message(state)
    intent = state["current_intent"]
    awaiting = state["awaiting_field"]

    # Fast path: off-topic (avoid LLM call for non-domain messages)
    if intent == Intent.OFF_TOPIC:
        reply = _OFF_TOPIC_RESPONSE
    elif awaiting and not state["lead_captured"]:
        # We're mid-lead-collection — prompt for the next field
        reply = _LEAD_PROMPT_TEMPLATES.get(
            awaiting,
            "Could you share a bit more about yourself so we can get you started?",
        )
    else:
        # Full Aria response
        if not ANTHROPIC_API_KEY:
            reply = (
                "Hi! I'm Aria from AutoStream. "
                "(API key not set — running in demo mode.)"
            )
        else:
            system = _build_aria_system(state)
            user_prompt = _history_to_user_prompt(messages[:-1], latest)
            reply = _llm(system, user_prompt, max_tokens=512)

    # Append Aria's reply to conversation history
    updated_messages = list(messages) + [{"role": "assistant", "content": reply}]
    return {"messages": updated_messages}


# ─── Node 4: collect_lead_node ────────────────────────────────────────────────

_FIELD_PATTERNS = {
    "name":     [
        r"my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"i(?:'m| am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"call me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ],
    "email":    [
        r"[\w.+-]+@[\w-]+\.[\w.]+",
    ],
    "platform": [
        r"\b(youtube|instagram|tiktok|twitch|facebook|linkedin|twitter|x\.com)\b",
    ],
}

_FIELD_ORDER = ["name", "email", "platform"]


def _extract_fields_regex(text: str) -> Dict[str, Optional[str]]:
    """Try to extract lead fields from user text using regex.

    Returns a dict with keys "name", "email", "platform" (value=None if not found).
    """
    text_lower = text.lower()
    found: Dict[str, Optional[str]] = {f: None for f in _FIELD_ORDER}

    for field, patterns in _FIELD_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                found[field] = m.group(0) if field == "email" else m.group(1)
                break

    return found


def _extract_fields_llm(user_message: str, awaiting_field: str) -> Dict[str, Optional[str]]:
    """Use a lightweight LLM call to extract any/all lead fields from the message.

    This is the partial recovery mechanism: even if the agent was asking for
    'email', if the user says 'I'm Arjun', we extract 'name' correctly.
    """
    system = (
        "Extract contact details from the user message. "
        "Return ONLY a JSON object with keys: name, email, platform. "
        "Set a key to null if the value is not present. "
        "platform should be one of: YouTube, Instagram, TikTok, Twitch, Facebook, or Other. "
        'Example: {"name": "Alex", "email": "alex@example.com", "platform": null}'
    )
    try:
        raw = _llm(system, user_message, max_tokens=128)
        # Strip fences
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "name":     data.get("name") or None,
                "email":    data.get("email") or None,
                "platform": data.get("platform") or None,
            }
    except Exception as exc:
        print(f"[nodes] LLM field extraction failed: {exc}")

    # Fallback to regex if LLM call fails
    return _extract_fields_regex(user_message)


def _next_missing_field(lead_info: LeadInfo) -> Optional[str]:
    """Return the next field to collect, in strict order: name → email → platform."""
    for field in _FIELD_ORDER:
        if not lead_info.get(field):
            return field
    return None


# Email-invalid re-ask prompt injected into generate_response_node via awaiting_field
_INVALID_EMAIL_PROMPT = (
    "Hmm, that email doesn't look quite right! 😊 "
    "Could you double-check and share a valid email address? "
    "(e.g. yourname@example.com)"
)


def collect_lead_node(state: AgentState) -> dict:
    """Extract lead fields from the user's message and advance the collection sequence.

    Partial recovery: if the agent was asking for field X but the user supplied
    field Y (e.g., gave their name when asked for email), we capture Y and
    re-evaluate which field to ask for next.

    Email validation: if an email is extracted but fails format validation,
    the agent re-asks for a valid email instead of storing a bad address.

    Updates: lead_info, awaiting_field
    """
    message = _latest_user_message(state)
    lead_info: LeadInfo = dict(state["lead_info"])   # mutable copy
    messages = list(state["messages"])

    # Try fast regex first, fall back to LLM if API key available
    if ANTHROPIC_API_KEY:
        extracted = _extract_fields_llm(message, state.get("awaiting_field", "name"))
    else:
        extracted = _extract_fields_regex(message)

    # ── Email validation guard ────────────────────────────────────────────────
    # If an email was extracted but is invalid, reject it and re-ask.
    if extracted.get("email") and not validate_email(extracted["email"]):
        print(
            f"[nodes] collect_lead: invalid email rejected: {extracted['email']!r}"
        )
        logger.warning("Invalid email submitted: %r — asking again.", extracted["email"])
        extracted["email"] = None   # discard bad value
        # Inject polite re-ask as the next assistant message
        messages.append({"role": "assistant", "content": _INVALID_EMAIL_PROMPT})
        return {
            "messages": messages,
            "lead_info": lead_info,
            "awaiting_field": "email",  # stay on email
        }

    # ── Merge extracted values into lead_info (only fill gaps, never overwrite) ─
    for field in _FIELD_ORDER:
        if extracted.get(field) and not lead_info.get(field):
            lead_info[field] = extracted[field]

    # Determine which field to ask for next
    next_field = _next_missing_field(lead_info)

    print(
        f"[nodes] collect_lead: extracted={extracted}, "
        f"lead_info={lead_info}, next={next_field}"
    )

    return {
        "messages": messages,
        "lead_info": lead_info,
        "awaiting_field": next_field,
    }


# ─── Node 5: capture_lead_node ────────────────────────────────────────────────

def capture_lead_node(state: AgentState) -> dict:
    """Call mock_lead_capture when all 3 lead fields are filled.

    Safety contract
    ---------------
    Before invoking the tool, this node asserts that ALL three required fields
    are present and non-empty. If the assertion fires it means a routing bug
    reached capture_lead_node prematurely — this is logged as a critical warning
    so it is immediately visible in monitoring.

    On success: sets lead_captured=True and injects a warm success message.
    On failure: logs the error and lets the conversation continue gracefully.
    """
    lead = state["lead_info"]
    messages = list(state["messages"])

    # ── Guard assertion — deliberate, visible safety check ───────────────────
    all_fields_present = all([
        lead.get("name"),
        lead.get("email"),
        lead.get("platform"),
    ])
    if not all_fields_present:
        logger.warning(
            "[capture_lead_node] GUARD FIRED — tool called prematurely! "
            "lead_info=%s  awaiting_field=%s",
            lead, state.get("awaiting_field"),
        )
        print(
            "[nodes] WARNING: capture_lead_node called with incomplete lead_info. "
            f"lead={lead}"
        )
    assert all_fields_present, (
        "Tool called prematurely — all fields must be collected first. "
        f"Missing: {[f for f in ('name','email','platform') if not lead.get(f)]}"
    )

    try:
        result = mock_lead_capture(
            name=lead["name"],
            email=lead["email"],
            platform=lead["platform"],
        )
        # result is now a plain dict: {status, lead_id, message}
        success_msg = (
            f"🎉 You're all set, {lead['name']}! "
            f"I've passed your details to our team (ref: {result['lead_id']}) — "
            f"expect a welcome email at {lead['email']} very soon. "
            f"Is there anything else about AutoStream I can help you with?"
        )
        messages.append({"role": "assistant", "content": success_msg})
        return {
            "messages": messages,
            "lead_captured": True,
            "awaiting_field": None,
        }

    except ValueError as exc:
        logger.error("[capture_lead_node] Validation error: %s", exc)
        print(f"[nodes] Lead capture validation error: {exc}")
        return {}   # Continue without marking captured; let generate_response handle it
    except Exception as exc:
        logger.exception("[capture_lead_node] Unexpected error: %s", exc)
        print(f"[nodes] Lead capture unexpected error: {exc}")
        return {}

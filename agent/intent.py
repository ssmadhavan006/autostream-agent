"""
agent/intent.py

Intent classification engine for AutoStream Agent.

Architecture:
  - Intent enum + IntentResult Pydantic model define the schema.
  - classify_intent() makes a structured LLM call (Claude via langchain-anthropic)
    and parses the JSON response safely.
  - apply_transition_rules() enforces stateful business logic on top of LLM output:
      * HIGH_INTENT_LEAD is sticky — don't lose a hot lead due to an off-topic detour.
      * Low-confidence results (< INTENT_MIN_CONFIDENCE) fall back to PRODUCT_INQUIRY.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, field_validator

from config.settings import (
    ANTHROPIC_API_KEY,
    INTENT_CLASSIFIER_TEMPERATURE,
    INTENT_CONTEXT_TURNS,
    INTENT_MIN_CONFIDENCE,
    LLM_BACKEND,
    LLM_MODEL,
)


# ─── Schema ───────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT_LEAD = "high_intent_lead"
    OFF_TOPIC = "off_topic"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float          # 0.0 – 1.0
    reasoning: str             # Why this intent was chosen

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, round(v, 4)))


# ─── Conversation turn helper ─────────────────────────────────────────────────

# A turn is (role, content) where role ∈ {"user", "assistant"}
ConversationTurn = Tuple[str, str]


# ─── System Prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an intent classification engine for AutoStream, a video streaming SaaS platform.

Your job is to analyse the user's latest message (and recent conversation context) and \
return a JSON object with EXACTLY these three fields:
  - "intent"     : one of ["greeting", "product_inquiry", "high_intent_lead", "off_topic"]
  - "confidence" : a float between 0.0 and 1.0
  - "reasoning"  : a short sentence (max 20 words) explaining your choice

Intent definitions:
  greeting          — The user is opening a conversation (hi, hello, hey, good morning, etc.)
  product_inquiry   — The user is asking about features, pricing, plans, limits, refunds, \
support, or how AutoStream works.
  high_intent_lead  — The user shows clear purchase or sign-up intent. Signals include:
                        * phrases like "want to try", "sign up", "get started", "subscribe", \
"buy", "purchase", "upgrade", "how do I join"
                        * mentioning a specific platform they stream to (YouTube, Instagram, \
TikTok, Twitch)
                        * asking about trials, onboarding, payment, or getting started
  off_topic         — The user's message is unrelated to AutoStream or video streaming \
(weather, jokes, coding help, etc.)

Rules:
  - Return ONLY valid JSON — no markdown fences, no preamble, no explanation outside the JSON.
  - If you are unsure, set confidence below 0.5 and choose the closest intent.
  - Never return an intent not in the list above.

Example output:
{"intent": "high_intent_lead", "confidence": 0.95, "reasoning": "User explicitly said they want to sign up today."}
"""


def _build_user_prompt(
    message: str,
    history: List[ConversationTurn],
) -> str:
    """Build the user-facing prompt including recent conversation context."""
    context_block = ""
    if history:
        recent = history[-INTENT_CONTEXT_TURNS:]
        lines = [f"  [{role.upper()}]: {content}" for role, content in recent]
        context_block = "Recent conversation context:\n" + "\n".join(lines) + "\n\n"

    return (
        f"{context_block}"
        f"Latest user message to classify:\n"
        f'  "{message}"\n\n'
        "Return JSON only."
    )


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str) -> str:
    """Call the configured LLM backend and return the raw text response.

    Uses the factory so the same backend switch (LLM_BACKEND env var) that
    controls response generation also controls intent classification.
    """
    from agent.llm_factory import get_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_chat_model(
        temperature=INTENT_CLASSIFIER_TEMPERATURE,
        max_tokens=256,
    )
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content


def _parse_llm_response(raw: str) -> IntentResult:
    """Parse the LLM's raw text response into an IntentResult.

    Handles:
      - Clean JSON strings
      - JSON wrapped in markdown fences (```json ... ```)
      - Extra whitespace or trailing characters
    """
    # Strip markdown fences if present
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Extract the first JSON object found (robust against stray text)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in LLM response: {raw!r}")

    data = json.loads(json_match.group())
    return IntentResult(**data)


# ─── Transition Rules ─────────────────────────────────────────────────────────

def apply_transition_rules(
    new_result: IntentResult,
    previous_intent: Optional[Intent] = None,
) -> IntentResult:
    """Apply stateful business logic on top of the raw LLM classification.

    Rules (applied in order):
      1. Confidence < INTENT_MIN_CONFIDENCE → fall back to PRODUCT_INQUIRY.
      2. If previous intent was HIGH_INTENT_LEAD and new intent is OFF_TOPIC,
         stay in HIGH_INTENT_LEAD — never lose a hot lead.

    Args:
        new_result:       The raw IntentResult from the LLM.
        previous_intent:  The intent of the immediately preceding turn, if any.

    Returns:
        Possibly-modified IntentResult with updated intent and reasoning.
    """
    result = new_result

    # Rule 1: Low-confidence fallback
    if result.confidence < INTENT_MIN_CONFIDENCE:
        result = IntentResult(
            intent=Intent.PRODUCT_INQUIRY,
            confidence=result.confidence,
            reasoning=(
                f"Low confidence ({result.confidence:.2f}); "
                "defaulting to product_inquiry as safe fallback."
            ),
        )

    # Rule 2: Sticky HIGH_INTENT_LEAD — don't drop a hot lead on an off-topic detour
    if (
        previous_intent == Intent.HIGH_INTENT_LEAD
        and result.intent == Intent.OFF_TOPIC
    ):
        result = IntentResult(
            intent=Intent.HIGH_INTENT_LEAD,
            confidence=max(result.confidence, 0.6),
            reasoning=(
                "User was in high-intent state; "
                "maintaining lead qualification despite off-topic message."
            ),
        )

    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def classify_intent(
    message: str,
    history: Optional[List[ConversationTurn]] = None,
    previous_intent: Optional[Intent] = None,
) -> IntentResult:
    """Classify the intent of a user message using a structured LLM call.

    Args:
        message:          The latest user message to classify.
        history:          Previous conversation turns [(role, content), ...].
                          The last INTENT_CONTEXT_TURNS turns are used as context.
        previous_intent:  The intent of the immediately preceding turn (for
                          transition rule enforcement).

    Returns:
        IntentResult with intent, confidence, and reasoning.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
        ValueError:       If the LLM returns an unparseable response.
    """
    if not ANTHROPIC_API_KEY and LLM_BACKEND == "anthropic":
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Either add your key to .env, or switch to Ollama by setting "
            "LLM_BACKEND=ollama in your .env file."
        )

    history = history or []
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _build_user_prompt(message, history)

    raw_response = _call_llm(system_prompt, user_prompt)
    result = _parse_llm_response(raw_response)
    result = apply_transition_rules(result, previous_intent=previous_intent)
    return result

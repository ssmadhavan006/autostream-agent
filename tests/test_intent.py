"""
tests/test_intent.py

Tests for the AutoStream intent classification engine.

Strategy:
  - All tests that would call the real LLM are patched via unittest.mock.patch,
    so they run without an ANTHROPIC_API_KEY and stay fast.
  - The transition-rule tests operate purely on Pydantic models — no LLM needed.
  - The parser tests validate the JSON extraction logic in isolation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.intent import (
    Intent,
    IntentResult,
    _build_user_prompt,
    _parse_llm_response,
    apply_transition_rules,
    classify_intent,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_raw(intent: str, confidence: float, reasoning: str) -> str:
    """Produce a raw JSON string as the LLM would return it."""
    return json.dumps({"intent": intent, "confidence": confidence, "reasoning": reasoning})


def _mock_classify(message: str, intent_str: str, confidence: float = 0.92,
                   history=None, previous_intent=None) -> IntentResult:
    """Run classify_intent with the LLM call mocked to return a specific intent."""
    raw = _make_raw(intent_str, confidence, f"Mocked: {intent_str}")
    mock_response = MagicMock()
    mock_response.content = raw

    with patch("agent.intent.ANTHROPIC_API_KEY", "sk-fake-key-for-testing"), \
         patch("agent.intent._call_llm", return_value=raw):
        return classify_intent(message, history=history, previous_intent=previous_intent)


# ─── Test 1: Greeting ─────────────────────────────────────────────────────────

def test_greeting_intent():
    """'Hello there!' should be classified as GREETING."""
    result = _mock_classify("Hello there!", "greeting")

    assert result.intent == Intent.GREETING
    assert result.confidence > 0.0
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0


# ─── Test 2: Product Inquiry ──────────────────────────────────────────────────

def test_product_inquiry_intent():
    """'What is the difference between Basic and Pro?' → PRODUCT_INQUIRY."""
    result = _mock_classify(
        "What is the difference between the Basic and Pro plans?",
        "product_inquiry",
    )

    assert result.intent == Intent.PRODUCT_INQUIRY
    assert result.confidence > 0.0


# ─── Test 3: High-Intent Lead ─────────────────────────────────────────────────

def test_high_intent_lead_sign_up():
    """'I want to sign up for the Pro plan today.' → HIGH_INTENT_LEAD."""
    result = _mock_classify(
        "I want to sign up for the Pro plan today.",
        "high_intent_lead",
    )

    assert result.intent == Intent.HIGH_INTENT_LEAD
    assert result.confidence >= 0.5


def test_high_intent_lead_platform_mention():
    """Mentioning YouTube + getting started → HIGH_INTENT_LEAD."""
    result = _mock_classify(
        "I run a YouTube channel and I'd love to get started with AutoStream.",
        "high_intent_lead",
    )

    assert result.intent == Intent.HIGH_INTENT_LEAD


def test_high_intent_lead_trial():
    """'Can I try AutoStream before buying?' → HIGH_INTENT_LEAD."""
    result = _mock_classify(
        "Is there a way to try AutoStream before I commit to a subscription?",
        "high_intent_lead",
    )

    assert result.intent == Intent.HIGH_INTENT_LEAD


# ─── Test 4: Off-Topic ────────────────────────────────────────────────────────

def test_off_topic_intent():
    """'What is the capital of France?' → OFF_TOPIC."""
    result = _mock_classify(
        "What is the capital of France?",
        "off_topic",
    )

    assert result.intent == Intent.OFF_TOPIC


# ─── Test 5: Transition Rule — Low Confidence Fallback ───────────────────────

def test_low_confidence_fallback_to_product_inquiry():
    """confidence < 0.5 must be demoted to PRODUCT_INQUIRY regardless of raw intent."""
    # Simulate LLM returning off_topic with very low confidence
    raw_result = IntentResult(
        intent=Intent.OFF_TOPIC,
        confidence=0.3,
        reasoning="Not sure what the user wants.",
    )

    final = apply_transition_rules(raw_result, previous_intent=None)

    assert final.intent == Intent.PRODUCT_INQUIRY, (
        f"Expected PRODUCT_INQUIRY fallback, got {final.intent}"
    )
    assert "fallback" in final.reasoning.lower()


def test_exact_boundary_confidence_not_demoted():
    """confidence == 0.5 must NOT be demoted (threshold is strictly <)."""
    raw_result = IntentResult(
        intent=Intent.GREETING,
        confidence=0.5,
        reasoning="User said hello.",
    )

    final = apply_transition_rules(raw_result, previous_intent=None)

    assert final.intent == Intent.GREETING, (
        "confidence == 0.5 should not trigger the low-confidence fallback"
    )


# ─── Test 6: Transition Rule — Sticky HIGH_INTENT_LEAD ───────────────────────

def test_sticky_high_intent_lead_blocks_off_topic_demotion():
    """If prev=HIGH_INTENT_LEAD and new=OFF_TOPIC, must stay HIGH_INTENT_LEAD."""
    raw_result = IntentResult(
        intent=Intent.OFF_TOPIC,
        confidence=0.85,
        reasoning="User asked something off-topic.",
    )

    final = apply_transition_rules(
        raw_result,
        previous_intent=Intent.HIGH_INTENT_LEAD,
    )

    assert final.intent == Intent.HIGH_INTENT_LEAD, (
        "HIGH_INTENT_LEAD must be sticky — off-topic should not drop the lead"
    )
    assert final.confidence >= 0.6


def test_sticky_rule_does_not_affect_non_lead_state():
    """If prev=GREETING and new=OFF_TOPIC, the result stays OFF_TOPIC."""
    raw_result = IntentResult(
        intent=Intent.OFF_TOPIC,
        confidence=0.8,
        reasoning="User is off-topic.",
    )

    final = apply_transition_rules(
        raw_result,
        previous_intent=Intent.GREETING,
    )

    assert final.intent == Intent.OFF_TOPIC


# ─── Test 7: IntentResult Pydantic Model ──────────────────────────────────────

def test_intent_result_confidence_clamped_above_one():
    """confidence > 1.0 must be clamped to 1.0."""
    result = IntentResult(
        intent=Intent.GREETING,
        confidence=1.5,
        reasoning="Over-confident LLM.",
    )
    assert result.confidence == 1.0


def test_intent_result_confidence_clamped_below_zero():
    """confidence < 0.0 must be clamped to 0.0."""
    result = IntentResult(
        intent=Intent.OFF_TOPIC,
        confidence=-0.2,
        reasoning="Negative confidence.",
    )
    assert result.confidence == 0.0


# ─── Test 8: JSON Parser robustness ───────────────────────────────────────────

def test_parser_clean_json():
    """_parse_llm_response handles a clean JSON string."""
    raw = _make_raw("product_inquiry", 0.88, "User asked about pricing.")
    result = _parse_llm_response(raw)
    assert result.intent == Intent.PRODUCT_INQUIRY
    assert result.confidence == 0.88


def test_parser_markdown_fenced_json():
    """_parse_llm_response handles JSON wrapped in ```json fences."""
    raw = '```json\n{"intent": "greeting", "confidence": 0.99, "reasoning": "User said hi."}\n```'
    result = _parse_llm_response(raw)
    assert result.intent == Intent.GREETING


def test_parser_json_with_preamble():
    """_parse_llm_response extracts JSON even when the LLM adds text before it."""
    raw = 'Here is the classification:\n{"intent": "high_intent_lead", "confidence": 0.91, "reasoning": "Sign-up intent."}'
    result = _parse_llm_response(raw)
    assert result.intent == Intent.HIGH_INTENT_LEAD


def test_parser_raises_on_no_json():
    """_parse_llm_response raises ValueError if no JSON object is present."""
    with pytest.raises(ValueError, match="No JSON object found"):
        _parse_llm_response("I cannot classify this message.")


# ─── Test 9: Context prompt builder ───────────────────────────────────────────

def test_prompt_includes_recent_history():
    """_build_user_prompt includes the latest conversation turns."""
    history = [
        ("user", "Hi there"),
        ("assistant", "Hello! How can I help you?"),
        ("user", "Tell me about pricing"),
        ("assistant", "We have Basic and Pro plans."),
    ]
    prompt = _build_user_prompt("What does Pro include?", history)

    # Should include context block
    assert "[USER]" in prompt or "USER" in prompt
    assert "Tell me about pricing" in prompt or "pricing" in prompt
    assert "What does Pro include?" in prompt


def test_prompt_no_history():
    """_build_user_prompt works fine with empty history."""
    prompt = _build_user_prompt("Hello!", [])
    assert "Hello!" in prompt
    assert "Return JSON only." in prompt


# ─── Test 10: classify_intent raises without API key ──────────────────────────

def test_classify_raises_without_api_key():
    """classify_intent must raise EnvironmentError if no API key is set."""
    with patch("agent.intent.ANTHROPIC_API_KEY", ""):
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            classify_intent("Hello!")


# ─── Test 11: classify_intent full mock round-trip (all 4 intents) ────────────

@pytest.mark.parametrize("message,intent_str", [
    ("Hey, good morning!", "greeting"),
    ("How much does the Pro plan cost?", "product_inquiry"),
    ("I want to subscribe to AutoStream right now!", "high_intent_lead"),
    ("What's 2 + 2?", "off_topic"),
])
def test_classify_all_four_intents(message, intent_str):
    """classify_intent correctly returns all four intent types when LLM is mocked."""
    result = _mock_classify(message, intent_str, confidence=0.9)
    assert result.intent == Intent(intent_str)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str)

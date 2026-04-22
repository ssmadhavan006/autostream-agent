"""
tests/test_graph.py

Integration-style tests for the AutoStream LangGraph state machine.

Strategy:
  - All LLM calls (_llm, classify_intent) are mocked so tests run without
    an ANTHROPIC_API_KEY and complete in < 1 second.
  - RAG retrieval is mocked to return predictable chunks.
  - mock_lead_capture is NOT mocked — it runs for real (writes to leads.jsonl).
  - Tests verify state transitions, routing, field collection, and partial recovery.
"""

from __future__ import annotations

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agent.intent import Intent, IntentResult
from agent.nodes import (
    _extract_fields_regex,
    _next_missing_field,
    capture_lead_node,
    classify_intent_node,
    collect_lead_node,
    generate_response_node,
    retrieve_context_node,
)
from agent.state import AgentState, LeadInfo, initial_state
from agent.tools import LeadCaptureResult, mock_lead_capture
from agent.graph import graph, intent_router, lead_router, post_context_router


# ─── Test helpers ─────────────────────────────────────────────────────────────

def _state_with_message(
    content: str,
    intent: Intent = Intent.GREETING,
    confidence: float = 0.9,
    lead_info: Optional[LeadInfo] = None,
    awaiting_field: Optional[str] = None,
    lead_captured: bool = False,
    rag_context: str = "",
    session_id: str = "test-session-001",
) -> AgentState:
    """Build an AgentState with one user message already in the history."""
    state = initial_state(session_id=session_id)
    state["messages"] = [{"role": "user", "content": content}]
    state["current_intent"] = intent
    state["intent_confidence"] = confidence
    state["lead_info"] = lead_info or LeadInfo(name=None, email=None, platform=None)
    state["awaiting_field"] = awaiting_field
    state["lead_captured"] = lead_captured
    state["rag_context"] = rag_context
    return state


def _mock_intent(intent: Intent, confidence: float = 0.9) -> IntentResult:
    return IntentResult(
        intent=intent,
        confidence=confidence,
        reasoning=f"Mocked: {intent.value}",
    )


# ─── AgentState / initial_state ───────────────────────────────────────────────

class TestInitialState:
    def test_all_fields_present(self):
        state = initial_state()
        assert "messages" in state
        assert "current_intent" in state
        assert "lead_info" in state
        assert "lead_captured" in state
        assert "rag_context" in state
        assert "awaiting_field" in state
        assert "session_id" in state
        assert "intent_confidence" in state

    def test_lead_info_starts_empty(self):
        state = initial_state()
        li = state["lead_info"]
        assert li["name"] is None
        assert li["email"] is None
        assert li["platform"] is None

    def test_lead_captured_starts_false(self):
        assert initial_state()["lead_captured"] is False

    def test_session_id_is_string(self):
        s = initial_state()
        assert isinstance(s["session_id"], str)
        assert len(s["session_id"]) > 0

    def test_custom_session_id(self):
        s = initial_state(session_id="custom-123")
        assert s["session_id"] == "custom-123"


# ─── classify_intent_node ─────────────────────────────────────────────────────

class TestClassifyIntentNode:
    def test_updates_intent_fields(self):
        state = _state_with_message("Hello!")
        with patch("agent.nodes.classify_intent",
                   return_value=_mock_intent(Intent.GREETING)):
            updates = classify_intent_node(state)
        assert updates["current_intent"] == Intent.GREETING
        assert updates["intent_confidence"] == 0.9

    def test_empty_message_returns_no_updates(self):
        state = initial_state()  # no messages
        updates = classify_intent_node(state)
        assert updates == {}

    def test_fallback_when_no_api_key(self):
        state = _state_with_message("Tell me about pricing")
        with patch("agent.nodes.ANTHROPIC_API_KEY", ""), \
             patch("agent.nodes.classify_intent",
                   side_effect=EnvironmentError("No key")):
            updates = classify_intent_node(state)
        assert updates["current_intent"] == Intent.PRODUCT_INQUIRY
        assert updates["intent_confidence"] == 0.5

    def test_passes_previous_intent_to_classifier(self):
        state = _state_with_message("Anyway...", intent=Intent.HIGH_INTENT_LEAD)
        captured = {}

        def fake_classify(message, history, previous_intent):
            captured["prev"] = previous_intent
            return _mock_intent(Intent.OFF_TOPIC)

        with patch("agent.nodes.classify_intent", side_effect=fake_classify):
            classify_intent_node(state)

        assert captured["prev"] == Intent.HIGH_INTENT_LEAD


# ─── retrieve_context_node ────────────────────────────────────────────────────

class TestRetrieveContextNode:
    def _fake_doc(self, content: str):
        from langchain.schema import Document
        return Document(page_content=content, metadata={"source": "test", "chunk_id": 0})

    def test_returns_chunks_as_rag_context(self):
        from rag.retriever import RetrievalResult
        state = _state_with_message("What is Pro plan price?",
                                    intent=Intent.PRODUCT_INQUIRY)
        fake_results = [
            RetrievalResult(document=self._fake_doc("Pro plan is $79/month."), score=0.8),
        ]
        with patch("agent.nodes.get_vectorstore"), \
             patch("agent.nodes.retrieve", return_value=fake_results):
            updates = retrieve_context_node(state)
        assert "79" in updates["rag_context"] or "Pro" in updates["rag_context"]

    def test_no_info_signal_clears_context(self):
        from rag.retriever import NoInfoSignal
        state = _state_with_message("What is nitrogen?")
        signal = NoInfoSignal(query="What is nitrogen?")
        with patch("agent.nodes.get_vectorstore"), \
             patch("agent.nodes.retrieve", return_value=signal):
            updates = retrieve_context_node(state)
        assert updates["rag_context"] == ""

    def test_exception_clears_context(self):
        state = _state_with_message("Pro plan price?")
        with patch("agent.nodes.get_vectorstore", side_effect=Exception("FAISS error")):
            updates = retrieve_context_node(state)
        assert updates["rag_context"] == ""


# ─── generate_response_node ───────────────────────────────────────────────────

class TestGenerateResponseNode:
    def test_off_topic_uses_canned_response(self):
        state = _state_with_message("What's 2+2?", intent=Intent.OFF_TOPIC)
        updates = generate_response_node(state)
        last_msg = updates["messages"][-1]
        assert last_msg["role"] == "assistant"
        assert "AutoStream" in last_msg["content"]

    def test_awaiting_name_prompts_for_name(self):
        state = _state_with_message("I want to sign up", awaiting_field="name")
        updates = generate_response_node(state)
        last_msg = updates["messages"][-1]["content"].lower()
        assert "name" in last_msg

    def test_awaiting_email_prompts_for_email(self):
        state = _state_with_message("Here", awaiting_field="email")
        updates = generate_response_node(state)
        last_msg = updates["messages"][-1]["content"].lower()
        assert "email" in last_msg

    def test_awaiting_platform_prompts_for_platform(self):
        state = _state_with_message("Ok", awaiting_field="platform")
        updates = generate_response_node(state)
        last_msg = updates["messages"][-1]["content"].lower()
        assert "platform" in last_msg or "youtube" in last_msg or "stream" in last_msg

    def test_llm_called_for_greeting(self):
        state = _state_with_message("Hello!", intent=Intent.GREETING)
        with patch("agent.nodes.ANTHROPIC_API_KEY", "fake-key"), \
             patch("agent.nodes._llm", return_value="Hi there! I'm Aria."):
            updates = generate_response_node(state)
        assert "Aria" in updates["messages"][-1]["content"]

    def test_no_api_key_returns_demo_message(self):
        state = _state_with_message("Hello!", intent=Intent.GREETING)
        with patch("agent.nodes.ANTHROPIC_API_KEY", ""):
            updates = generate_response_node(state)
        assert updates["messages"][-1]["role"] == "assistant"
        assert "demo" in updates["messages"][-1]["content"].lower()


# ─── collect_lead_node ────────────────────────────────────────────────────────

class TestCollectLeadNode:
    def test_extracts_email_from_message(self):
        state = _state_with_message(
            "Sure, my email is alice@example.com",
            intent=Intent.HIGH_INTENT_LEAD,
            awaiting_field="email",
        )
        with patch("agent.nodes.ANTHROPIC_API_KEY", ""), \
             patch("agent.nodes._extract_fields_llm",
                   return_value={"name": None, "email": "alice@example.com", "platform": None}):
            updates = collect_lead_node(state)
        assert updates["lead_info"]["email"] == "alice@example.com"

    def test_partial_recovery_name_while_awaiting_email(self):
        """If agent asked for email but user gives name, we extract name anyway."""
        state = _state_with_message(
            "Oh wait, my name is Arjun",
            intent=Intent.HIGH_INTENT_LEAD,
            awaiting_field="email",
        )
        with patch("agent.nodes.ANTHROPIC_API_KEY", ""), \
             patch("agent.nodes._extract_fields_llm",
                   return_value={"name": "Arjun", "email": None, "platform": None}):
            updates = collect_lead_node(state)
        # Name extracted despite asking for email
        assert updates["lead_info"]["name"] == "Arjun"
        # Still needs email → awaiting_field stays on "email"
        assert updates["awaiting_field"] == "email"

    def test_next_field_advances_to_email_after_name(self):
        state = _state_with_message(
            "I'm Alice",
            intent=Intent.HIGH_INTENT_LEAD,
            lead_info=LeadInfo(name=None, email=None, platform=None),
            awaiting_field="name",
        )
        with patch("agent.nodes._extract_fields_llm",
                   return_value={"name": "Alice", "email": None, "platform": None}):
            updates = collect_lead_node(state)
        assert updates["awaiting_field"] == "email"

    def test_awaiting_none_when_all_fields_filled(self):
        state = _state_with_message(
            "I use YouTube",
            intent=Intent.HIGH_INTENT_LEAD,
            lead_info=LeadInfo(name="Alice", email="alice@x.com", platform=None),
            awaiting_field="platform",
        )
        with patch("agent.nodes._extract_fields_llm",
                   return_value={"name": None, "email": None, "platform": "YouTube"}):
            updates = collect_lead_node(state)
        assert updates["lead_info"]["platform"] == "YouTube"
        assert updates["awaiting_field"] is None

    def test_regex_extracts_email_fallback(self):
        """_extract_fields_regex must find emails correctly."""
        extracted = _extract_fields_regex("Reach me at test.user+filter@domain.co.uk")
        assert extracted["email"] == "test.user+filter@domain.co.uk"

    def test_regex_extracts_platform(self):
        extracted = _extract_fields_regex("I mainly post on TikTok")
        assert extracted["platform"] is not None
        assert "tiktok" in extracted["platform"].lower()


# ─── capture_lead_node ────────────────────────────────────────────────────────

class TestCaptureLeadNode:
    def test_sets_lead_captured_true(self):
        state = _state_with_message(
            "YouTube",
            intent=Intent.HIGH_INTENT_LEAD,
            lead_info=LeadInfo(name="Bob", email="bob@test.com", platform="YouTube"),
            session_id="test-capture-001",
        )
        updates = capture_lead_node(state)
        assert updates.get("lead_captured") is True

    def test_appends_success_message_to_history(self):
        state = _state_with_message(
            "YouTube",
            intent=Intent.HIGH_INTENT_LEAD,
            lead_info=LeadInfo(name="Carol", email="carol@test.com", platform="Instagram"),
            session_id="test-capture-002",
        )
        updates = capture_lead_node(state)
        last = updates["messages"][-1]
        assert last["role"] == "assistant"
        assert "Carol" in last["content"]

    def test_raises_assertion_on_missing_field(self):
        """Guard assertion fires when name is empty — this is the correct behaviour."""
        state = _state_with_message(
            "",
            lead_info=LeadInfo(name="", email="x@y.com", platform="TikTok"),
            session_id="test-capture-003",
        )
        # Phase 4: capture_lead_node raises AssertionError when fields are incomplete
        with pytest.raises(AssertionError, match="prematurely"):
            capture_lead_node(state)


# ─── Router functions ─────────────────────────────────────────────────────────

class TestRouters:
    def test_intent_router_greeting_goes_to_generate(self):
        state = _state_with_message("Hi", intent=Intent.GREETING)
        assert intent_router(state) == "generate_response"

    def test_intent_router_off_topic_goes_to_generate(self):
        state = _state_with_message("Tell me a joke", intent=Intent.OFF_TOPIC)
        assert intent_router(state) == "generate_response"

    def test_intent_router_product_inquiry_needs_context(self):
        state = _state_with_message("Pricing?", intent=Intent.PRODUCT_INQUIRY)
        assert intent_router(state) == "need_context"

    def test_intent_router_high_intent_needs_context(self):
        state = _state_with_message("Sign me up!", intent=Intent.HIGH_INTENT_LEAD)
        assert intent_router(state) == "need_context"

    def test_post_context_router_inquiry_goes_to_generate(self):
        state = _state_with_message("Pricing?", intent=Intent.PRODUCT_INQUIRY)
        assert post_context_router(state) == "generate_response"

    def test_post_context_router_lead_not_captured_goes_to_collect(self):
        state = _state_with_message("Sign up!", intent=Intent.HIGH_INTENT_LEAD,
                                    lead_captured=False)
        assert post_context_router(state) == "collect_lead"

    def test_post_context_router_already_captured_goes_to_generate(self):
        state = _state_with_message("Thanks!", intent=Intent.HIGH_INTENT_LEAD,
                                    lead_captured=True)
        assert post_context_router(state) == "generate_response"

    def test_lead_router_incomplete_returns_respond(self):
        state = _state_with_message(
            "", lead_info=LeadInfo(name="Alice", email=None, platform=None))
        assert lead_router(state) == "respond"

    def test_lead_router_complete_returns_capture(self):
        state = _state_with_message(
            "", lead_info=LeadInfo(name="Alice", email="a@b.com", platform="YouTube"))
        assert lead_router(state) == "capture"


# ─── Graph compile check ──────────────────────────────────────────────────────

class TestGraphCompile:
    def test_graph_compiles_without_error(self):
        from agent.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_has_all_nodes(self):
        from agent.graph import graph
        # CompiledStateGraph stores nodes in .nodes dict
        node_names = set(graph.nodes.keys())
        expected = {
            "classify_intent",
            "retrieve_context",
            "collect_lead",
            "capture_lead",
            "generate_response",
        }
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )


# ─── 6-turn state persistence simulation ─────────────────────────────────────

class TestSixTurnSimulation:
    """Simulate a 6-turn conversation purely in memory (no LLM, no FAISS)."""

    def _run_turn(self, state: AgentState, user_msg: str,
                  mock_intent: Intent) -> AgentState:
        """Run one turn through the graph with LLM and RAG mocked."""
        from rag.retriever import NoInfoSignal
        messages = list(state["messages"]) + [{"role": "user", "content": user_msg}]
        input_state = dict(state)
        input_state["messages"] = messages

        with patch("agent.nodes.classify_intent",
                   return_value=_mock_intent(mock_intent)), \
             patch("agent.nodes.get_vectorstore"), \
             patch("agent.nodes.retrieve",
                   return_value=NoInfoSignal(query=user_msg)), \
             patch("agent.nodes.ANTHROPIC_API_KEY", "fake-key"), \
             patch("agent.nodes._llm", return_value="Mocked Aria reply"), \
             patch("agent.nodes._extract_fields_llm",
                   side_effect=lambda msg, _: _extract_fields_regex(msg)):
            return graph.invoke(input_state)

    def test_state_persists_across_six_turns(self):
        """State (especially lead_info) accumulates correctly over 6 turns."""
        state = initial_state(session_id="sim-6turn")

        # Turn 1: greeting
        state = self._run_turn(state, "Hi!", Intent.GREETING)
        assert len(state["messages"]) == 2   # user + assistant
        assert state["current_intent"] == Intent.GREETING

        # Turn 2: product inquiry
        state = self._run_turn(state, "What does Pro plan include?", Intent.PRODUCT_INQUIRY)
        assert len(state["messages"]) == 4
        assert state["current_intent"] == Intent.PRODUCT_INQUIRY

        # Turn 3: high intent — triggers lead collection (awaiting name)
        state = self._run_turn(state, "I want to sign up!", Intent.HIGH_INTENT_LEAD)
        assert state["current_intent"] == Intent.HIGH_INTENT_LEAD
        # awaiting_field should be "name" since none collected yet
        assert state["awaiting_field"] == "name"

        # Turn 4: provide name
        state = self._run_turn(state, "My name is Diana", Intent.HIGH_INTENT_LEAD)
        # Name should be captured from regex
        assert state["lead_info"]["name"] == "Diana"
        assert state["awaiting_field"] == "email"

        # Turn 5: provide email
        state = self._run_turn(state, "diana@example.com", Intent.HIGH_INTENT_LEAD)
        assert state["lead_info"]["email"] == "diana@example.com"
        assert state["awaiting_field"] == "platform"

        # Turn 6: provide platform — triggers capture
        state = self._run_turn(state, "I use YouTube mainly", Intent.HIGH_INTENT_LEAD)
        assert "youtube" in (state["lead_info"]["platform"] or "").lower()
        assert state["lead_captured"] is True
        assert len(state["messages"]) >= 12

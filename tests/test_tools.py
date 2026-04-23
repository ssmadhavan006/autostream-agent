"""
tests/test_tools.py

Tests for agent/tools.py:
  - mock_lead_capture (core function)
  - validate_email (email format guard)
  - LeadCaptureInput Pydantic schema
  - lead_capture_tool StructuredTool wrapper
  - Guard assertion in capture_lead_node (nodes.py)
  - Email re-ask flow in collect_lead_node (nodes.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from agent.tools import (
    LeadCaptureInput,
    LeadCaptureResult,
    lead_capture_tool,
    mock_lead_capture,
    validate_email,
)
from agent.nodes import capture_lead_node, collect_lead_node
from agent.state import AgentState, LeadInfo, initial_state


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _state_with_lead(
    lead_info: LeadInfo,
    awaiting_field: Optional[str] = None,
    user_msg: str = "",
    session_id: str = "test-tools-session",
) -> AgentState:
    state = initial_state(session_id=session_id)
    state["lead_info"] = lead_info
    state["awaiting_field"] = awaiting_field
    if user_msg:
        state["messages"] = [{"role": "user", "content": user_msg}]
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — validate_email
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateEmail:
    def test_valid_simple_email(self):
        assert validate_email("alice@example.com") is True

    def test_valid_with_plus_alias(self):
        assert validate_email("alice+filter@example.com") is True

    def test_valid_subdomain(self):
        assert validate_email("user@mail.example.co.uk") is True

    def test_valid_dots_in_local(self):
        assert validate_email("first.last@company.org") is True

    def test_invalid_missing_at(self):
        assert validate_email("notanemail.com") is False

    def test_invalid_missing_tld(self):
        assert validate_email("user@domain") is False

    def test_invalid_spaces(self):
        assert validate_email("user name@example.com") is False

    def test_invalid_double_at(self):
        assert validate_email("user@@example.com") is False

    def test_invalid_empty_string(self):
        assert validate_email("") is False

    def test_invalid_only_at(self):
        assert validate_email("@") is False

    def test_invalid_no_local_part(self):
        assert validate_email("@example.com") is False

    def test_case_insensitive(self):
        assert validate_email("Alice@EXAMPLE.COM") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — mock_lead_capture core function
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockLeadCapture:
    def test_successful_capture_returns_dict(self, tmp_path):
        """Successful capture returns a dict with status, lead_id, message."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            result = mock_lead_capture(
                name="Alice Smith",
                email="alice@example.com",
                platform="YouTube",
            )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "lead_id" in result
        assert result["lead_id"].startswith("LEAD-")
        assert "Alice" in result["message"]

    def test_lead_id_format(self, tmp_path):
        """Lead ID follows LEAD-NNNNN format derived from email hash."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            result = mock_lead_capture("Bob", "bob@test.com", "TikTok")
        import re
        assert re.match(r"LEAD-\d{5}", result["lead_id"])

    def test_lead_id_deterministic_for_same_email(self, tmp_path):
        """Same email always produces the same lead_id (hash-based)."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            r1 = mock_lead_capture("Alice", "alice@example.com", "YouTube")
            r2 = mock_lead_capture("Alice", "alice@example.com", "TikTok")
        assert r1["lead_id"] == r2["lead_id"]

    def test_raises_when_name_missing(self, tmp_path):
        """Raises ValueError if name is empty."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            with pytest.raises(ValueError, match="name"):
                mock_lead_capture(name="", email="a@b.com", platform="YouTube")

    def test_raises_when_email_missing(self, tmp_path):
        """Raises ValueError if email is empty."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            with pytest.raises(ValueError, match="email"):
                mock_lead_capture(name="Alice", email="", platform="YouTube")

    def test_raises_when_platform_missing(self, tmp_path):
        """Raises ValueError if platform is empty."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            with pytest.raises(ValueError, match="platform"):
                mock_lead_capture(name="Alice", email="a@b.com", platform="")

    def test_raises_on_invalid_email_format(self, tmp_path):
        """Raises ValueError if email fails format validation."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            with pytest.raises(ValueError, match="valid email"):
                mock_lead_capture(
                    name="Alice", email="not-valid-email", platform="YouTube"
                )

    def test_lead_persisted_to_jsonl(self, tmp_path):
        """Successful capture writes a JSON record to the JSONL file."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            mock_lead_capture("Carol", "carol@example.com", "Instagram")

        assert leads_file.exists()
        line = leads_file.read_text().strip()
        record = json.loads(line)
        assert record["name"] == "Carol"
        assert record["email"] == "carol@example.com"
        assert record["platform"] == "Instagram"

    def test_whitespace_trimmed_from_inputs(self, tmp_path):
        """Leading/trailing whitespace is stripped from all fields."""
        leads_file = tmp_path / "leads.jsonl"
        with patch("agent.tools._LEADS_FILE", leads_file):
            result = mock_lead_capture(
                "  Dave  ", "  dave@example.com  ", "  YouTube  "
            )
        assert result["status"] == "success"
        record = json.loads(leads_file.read_text().strip())
        assert record["name"] == "Dave"
        assert record["email"] == "dave@example.com"
        assert record["platform"] == "YouTube"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — LeadCaptureInput Pydantic schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeadCaptureInput:
    def test_valid_input_passes(self):
        inp = LeadCaptureInput(
            name="Eve", email="eve@example.com", platform="TikTok"
        )
        assert inp.name == "Eve"
        assert inp.email == "eve@example.com"
        assert inp.platform == "TikTok"

    def test_rejects_empty_name(self):
        with pytest.raises(Exception):   # pydantic.ValidationError
            LeadCaptureInput(name="", email="e@example.com", platform="YouTube")

    def test_rejects_empty_email(self):
        with pytest.raises(Exception):
            LeadCaptureInput(name="Eve", email="", platform="YouTube")

    def test_rejects_invalid_email(self):
        with pytest.raises(Exception):
            LeadCaptureInput(name="Eve", email="bad-email", platform="YouTube")

    def test_rejects_empty_platform(self):
        with pytest.raises(Exception):
            LeadCaptureInput(name="Eve", email="e@example.com", platform="")

    def test_whitespace_stripped_by_validator(self):
        inp = LeadCaptureInput(
            name="  Frank  ", email="frank@x.com", platform="  YouTube  "
        )
        assert inp.name == "Frank"
        assert inp.platform == "YouTube"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — StructuredTool wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeadCaptureTool:
    def test_tool_name_is_mock_lead_capture(self):
        assert lead_capture_tool.name == "mock_lead_capture"

    def test_tool_description_mentions_precondition(self):
        desc = lead_capture_tool.description
        assert "confirmed" in desc.lower() or "must" in desc.lower()

    def test_tool_has_args_schema(self):
        assert lead_capture_tool.args_schema is not None

    def test_tool_invocation_succeeds(self, tmp_path):
        """StructuredTool.invoke() calls mock_lead_capture and returns a dict."""
        with patch("agent.tools._LEADS_FILE", tmp_path / "leads.jsonl"):
            result = lead_capture_tool.invoke({
                "name": "Grace",
                "email": "grace@example.com",
                "platform": "YouTube",
            })
        assert isinstance(result, dict)
        assert result["status"] == "success"

    def test_tool_invocation_raises_on_missing_field(self, tmp_path):
        """Tool raises when a required field is missing/empty."""
        with patch("agent.tools._LEADS_FILE", tmp_path / "leads.jsonl"):
            with pytest.raises(Exception):
                lead_capture_tool.invoke({
                    "name": "",
                    "email": "grace@example.com",
                    "platform": "YouTube",
                })


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Guard assertion in capture_lead_node
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptureLeadGuard:
    def test_guard_fires_on_missing_name(self):
        """AssertionError raised when name is missing."""
        state = _state_with_lead(
            LeadInfo(name=None, email="a@b.com", platform="YouTube")
        )
        with pytest.raises(AssertionError, match="prematurely"):
            capture_lead_node(state)

    def test_guard_fires_on_missing_email(self):
        """AssertionError raised when email is missing."""
        state = _state_with_lead(
            LeadInfo(name="Alice", email=None, platform="YouTube")
        )
        with pytest.raises(AssertionError, match="prematurely"):
            capture_lead_node(state)

    def test_guard_fires_on_missing_platform(self):
        """AssertionError raised when platform is missing."""
        state = _state_with_lead(
            LeadInfo(name="Alice", email="a@b.com", platform=None)
        )
        with pytest.raises(AssertionError, match="prematurely"):
            capture_lead_node(state)

    def test_guard_does_not_fire_when_all_present(self, tmp_path):
        """No AssertionError when all three fields are filled."""
        state = _state_with_lead(
            LeadInfo(name="Alice", email="alice@example.com", platform="YouTube"),
            session_id="guard-test",
        )
        with patch("agent.tools._LEADS_FILE", tmp_path / "leads.jsonl"):
            updates = capture_lead_node(state)
        assert updates.get("lead_captured") is True

    def test_guard_message_lists_missing_fields(self):
        """AssertionError message names the specific missing fields."""
        state = _state_with_lead(
            LeadInfo(name="Alice", email=None, platform=None)
        )
        with pytest.raises(AssertionError) as exc_info:
            capture_lead_node(state)
        msg = str(exc_info.value)
        assert "email" in msg or "platform" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Email validation re-ask in collect_lead_node
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmailReAskFlow:
    def test_invalid_email_rejected_and_reask_injected(self):
        """If regex extracts an invalid email, it is discarded and agent re-asks."""
        state = _state_with_lead(
            LeadInfo(name="Bob", email=None, platform=None),
            awaiting_field="email",
            user_msg="my email is not-valid-email",
        )

        with patch("agent.nodes.is_llm_available", return_value=False), \
             patch("agent.nodes._extract_fields_regex",
                   return_value={"name": None, "email": "not-valid-email", "platform": None}):
            updates = collect_lead_node(state)

        # Email must NOT be stored
        assert updates["lead_info"]["email"] is None
        # Must still be awaiting email
        assert updates["awaiting_field"] == "email"
        # A polite re-ask message must have been injected
        last_msg = updates["messages"][-1]
        assert last_msg["role"] == "assistant"
        assert "email" in last_msg["content"].lower()

    def test_valid_email_accepted(self):
        """A valid email is accepted and stored in lead_info."""
        state = _state_with_lead(
            LeadInfo(name="Bob", email=None, platform=None),
            awaiting_field="email",
            user_msg="bob@example.com",
        )

        with patch("agent.nodes.is_llm_available", return_value=False), \
             patch("agent.nodes._extract_fields_regex",
                   return_value={"name": None, "email": "bob@example.com", "platform": None}):
            updates = collect_lead_node(state)

        assert updates["lead_info"]["email"] == "bob@example.com"
        assert updates["awaiting_field"] == "platform"

    def test_invalid_email_does_not_advance_field(self):
        """Invalid email must not advance awaiting_field beyond 'email'."""
        state = _state_with_lead(
            LeadInfo(name="Carol", email=None, platform=None),
            awaiting_field="email",
            user_msg="carol at gmail",
        )

        with patch("agent.nodes.is_llm_available", return_value=False), \
             patch("agent.nodes._extract_fields_regex",
                   return_value={"name": None, "email": "carol at gmail", "platform": None}):
            updates = collect_lead_node(state)

        # Still awaiting email — never advanced to platform
        assert updates.get("awaiting_field") == "email"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — LeadCaptureResult backward-compat wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeadCaptureResultCompat:
    def test_wraps_success_dict(self):
        r = LeadCaptureResult({"status": "success", "lead_id": "LEAD-12345", "message": "Hi!"})
        assert r.success is True
        assert r.lead_id == "LEAD-12345"
        assert r.message == "Hi!"

    def test_wraps_failure_dict(self):
        r = LeadCaptureResult({"status": "error", "lead_id": "", "message": "Fail"})
        assert r.success is False

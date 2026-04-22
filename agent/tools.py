"""
agent/tools.py

Tool definitions for AutoStream Agent.

mock_lead_capture   — Simulates writing a qualified lead to a CRM / database.
                      Exposed both as a plain Python function AND as a LangChain
                      StructuredTool so it can be bound to an LLM's tool-calling API.

Email validation    — validate_email() enforces RFC-5321-style email format using
                      regex before the field is accepted into lead_info.

Guard contract      — The StructuredTool docstring explicitly states the precondition:
                      all three fields (name, email, platform) MUST be confirmed before
                      this tool is called. The guard is also enforced programmatically
                      in capture_lead_node (agent/nodes.py).
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from langchain.tools import StructuredTool
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ─── Email Validation ─────────────────────────────────────────────────────────

# RFC-5321 simplified pattern: local@domain.tld
# Deliberately strict — catches fat-finger errors at collection time.
_EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
    re.IGNORECASE,
)


def validate_email(email: str) -> bool:
    """Return True if *email* matches a valid RFC-5321 email pattern.

    Examples
    --------
    >>> validate_email("alice@example.com")
    True
    >>> validate_email("not-an-email")
    False
    >>> validate_email("missing@tld")
    False
    >>> validate_email("spaces in@email.com")
    False
    """
    return bool(_EMAIL_REGEX.match(email.strip()))


# ─── Core function ────────────────────────────────────────────────────────────

_LEADS_FILE = Path(__file__).parent.parent / "transcripts" / "leads.jsonl"


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """Captures a qualified lead. Must only be called when all three fields are confirmed.

    This function simulates a CRM write. In production it would POST to a real API.

    Preconditions (enforced by the calling node):
      - name     : non-empty, the prospect's full name
      - email    : non-empty, valid email address (validated by validate_email())
      - platform : non-empty, the streaming platform (YouTube, TikTok, Instagram, etc.)

    DO NOT call this tool unless the agent has explicitly confirmed all three values
    with the user. Calling it prematurely will create a corrupt lead record.

    Args:
        name:     Full name of the prospect.
        email:    Verified email address.
        platform: Primary streaming platform.

    Returns:
        dict with keys: status, lead_id, message

    Raises:
        ValueError: If any required field is empty or email is invalid.
    """
    name = (name or "").strip()
    email = (email or "").strip()
    platform = (platform or "").strip()

    # ── Field presence checks ────────────────────────────────────────────────
    if not name:
        raise ValueError("Lead capture failed: 'name' is required.")
    if not email:
        raise ValueError("Lead capture failed: 'email' is required.")
    if not platform:
        raise ValueError("Lead capture failed: 'platform' is required.")

    # ── Email format validation ───────────────────────────────────────────────
    if not validate_email(email):
        raise ValueError(
            f"Lead capture failed: '{email}' is not a valid email address. "
            "Please provide a valid email (e.g. alice@example.com)."
        )

    # ── Generate deterministic-style lead ID ─────────────────────────────────
    lead_id = f"LEAD-{hash(email) % 100000:05d}"

    # ── Persist to JSONL ─────────────────────────────────────────────────────
    record: Dict[str, Any] = {
        "lead_id": lead_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "email": email,
        "platform": platform,
    }
    _LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_LEADS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Lead captured successfully: {name}, {email}, {platform}")
    logger.info("Lead captured: lead_id=%s name=%s email=%s", lead_id, name, email)

    return {
        "status": "success",
        "lead_id": lead_id,
        "message": f"Welcome aboard, {name}!",
    }


# ─── Pydantic input schema (required by StructuredTool) ──────────────────────

class LeadCaptureInput(BaseModel):
    """Input schema for the mock_lead_capture StructuredTool."""

    name: str
    email: str
    platform: str

    @field_validator("name", "email", "platform")
    @classmethod
    def must_not_be_empty(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"'{info.field_name}' must not be empty.")
        return v.strip()

    @field_validator("email")
    @classmethod
    def must_be_valid_email(cls, v: str) -> str:
        if not validate_email(v):
            raise ValueError(
                f"'{v}' is not a valid email address. "
                "Please provide a valid email (e.g. alice@example.com)."
            )
        return v


# ─── LangChain StructuredTool ─────────────────────────────────────────────────

lead_capture_tool = StructuredTool.from_function(
    func=mock_lead_capture,
    name="mock_lead_capture",
    description=(
        "Captures a qualified sales lead into the CRM. "
        "MUST only be called after the agent has collected AND confirmed "
        "all three fields from the user: name (str), email (valid email address), "
        "and platform (their primary streaming platform e.g. YouTube, TikTok). "
        "Never call this tool if any field is missing, empty, or unconfirmed."
    ),
    args_schema=LeadCaptureInput,
    return_direct=False,
)


# ─── Convenience re-export for backward compatibility ────────────────────────

class LeadCaptureResult:
    """Thin wrapper around the dict returned by mock_lead_capture.

    Kept for backward compatibility with code written in Phase 3.
    """

    def __init__(self, result_dict: dict):
        self.success = result_dict.get("status") == "success"
        self.lead_id = result_dict.get("lead_id", "")
        self.message = result_dict.get("message", "")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LeadCaptureResult(success={self.success}, "
            f"lead_id={self.lead_id!r}, message={self.message!r})"
        )

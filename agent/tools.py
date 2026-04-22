"""
agent/tools.py

Tool definitions for AutoStream Agent.

mock_lead_capture   — Simulates writing a captured lead to a CRM / database.
                      In production this would POST to a real API.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# ─── Lead Capture ─────────────────────────────────────────────────────────────

class LeadCaptureResult:
    """Return value from mock_lead_capture()."""

    def __init__(self, success: bool, lead_id: str, message: str):
        self.success = success
        self.lead_id = lead_id
        self.message = message

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LeadCaptureResult(success={self.success}, "
            f"lead_id={self.lead_id!r}, message={self.message!r})"
        )


_LEADS_FILE = Path(__file__).parent.parent / "transcripts" / "leads.jsonl"


def mock_lead_capture(
    name: str,
    email: str,
    platform: str,
    session_id: str = "unknown",
    **extra: Any,
) -> LeadCaptureResult:
    """Simulate capturing a qualified lead into a CRM.

    Writes the lead as a JSON line to transcripts/leads.jsonl so it persists
    across runs and can be inspected manually.

    Args:
        name:       Full name of the prospect.
        email:      Email address.
        platform:   Streaming platform (YouTube, TikTok, Instagram, etc.)
        session_id: The session this lead came from.
        **extra:    Any additional fields to store (forward-compatible).

    Returns:
        LeadCaptureResult with success=True and a generated lead_id.

    Raises:
        ValueError: If name, email, or platform is empty.
    """
    name = (name or "").strip()
    email = (email or "").strip()
    platform = (platform or "").strip()

    if not name:
        raise ValueError("Lead capture failed: 'name' is required.")
    if not email:
        raise ValueError("Lead capture failed: 'email' is required.")
    if not platform:
        raise ValueError("Lead capture failed: 'platform' is required.")

    # Generate a simple deterministic-ish lead ID
    timestamp = datetime.now(timezone.utc)
    lead_id = f"LEAD-{timestamp.strftime('%Y%m%d%H%M%S')}-{session_id[:8].upper()}"

    record: Dict[str, Any] = {
        "lead_id": lead_id,
        "session_id": session_id,
        "captured_at": timestamp.isoformat(),
        "name": name,
        "email": email,
        "platform": platform,
        **extra,
    }

    # Persist to JSONL file (append mode)
    _LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_LEADS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[tools] Lead captured: {lead_id} — {name} <{email}> ({platform})")

    return LeadCaptureResult(
        success=True,
        lead_id=lead_id,
        message=(
            f"Lead successfully captured for {name}. "
            f"Our team will reach out at {email} shortly."
        ),
    )

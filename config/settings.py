"""
config/settings.py

Central configuration for AutoStream Agent.
All environment variables and model constants are loaded here.
Import this module everywhere instead of calling os.getenv directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Load .env (silently skipped if file doesn't exist) ───────────────────────
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# ─── LLM & Embedding Models ───────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Intent Classifier Settings ───────────────────────────────────────────────
# How many previous conversation turns to include as context for intent detection
INTENT_CONTEXT_TURNS: int = int(os.getenv("INTENT_CONTEXT_TURNS", "3"))

# LLM temperature for intent classification (lower = more deterministic)
INTENT_CLASSIFIER_TEMPERATURE: float = float(
    os.getenv("INTENT_CLASSIFIER_TEMPERATURE", "0.0")
)

# Minimum confidence below which the classifier falls back to PRODUCT_INQUIRY
INTENT_MIN_CONFIDENCE: float = float(os.getenv("INTENT_MIN_CONFIDENCE", "0.5"))

# ─── RAG Settings ─────────────────────────────────────────────────────────────
FAISS_INDEX_DIR: Path = Path(__file__).parent.parent / "faiss_index"
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "300"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
RAG_CONFIDENCE_THRESHOLD: float = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.25"))

# ─── Transcript / Persistence ─────────────────────────────────────────────────
TRANSCRIPTS_DIR: Path = Path(__file__).parent.parent / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Sanity helpers ───────────────────────────────────────────────────────────

def require_api_key() -> str:
    """Return ANTHROPIC_API_KEY or raise a clear error if it's missing."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return ANTHROPIC_API_KEY

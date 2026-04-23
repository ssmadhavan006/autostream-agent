"""
agent/llm_factory.py

LLM backend factory for AutoStream Agent.

Supports two backends controlled by the LLM_BACKEND env var:
  - "anthropic"  (default) — Claude via Anthropic API (requires ANTHROPIC_API_KEY)
  - "ollama"               — Any model served by Ollama (free, local/cloud-routed)

Usage
-----
  from agent.llm_factory import get_chat_model, is_llm_available

  llm = get_chat_model(temperature=0.4, max_tokens=512)
  response = llm.invoke([SystemMessage(...), HumanMessage(...)])

Adding a new backend
--------------------
  Add a new elif branch in get_chat_model() and update is_llm_available() accordingly.
"""

from __future__ import annotations

from config.settings import (
    ANTHROPIC_API_KEY,
    LLM_BACKEND,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def get_chat_model(temperature: float = 0.4, max_tokens: int = 512):
    """Return a configured LangChain chat model based on the LLM_BACKEND setting.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens:  Maximum tokens in the response (ignored for Ollama, which
                     uses the model's own default).

    Returns:
        A LangChain BaseChatModel instance ready to .invoke().

    Raises:
        ValueError: If LLM_BACKEND is set to an unrecognised value.
    """
    if LLM_BACKEND == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            # num_predict is Ollama's equivalent of max_tokens
            num_predict=max_tokens,
        )

    elif LLM_BACKEND == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND={LLM_BACKEND!r}. "
            "Supported values: 'anthropic', 'ollama'."
        )


def is_llm_available() -> bool:
    """Return True if the configured LLM backend is ready to use.

    - ollama  : always True (Ollama needs no API key)
    - anthropic: True only when ANTHROPIC_API_KEY is set
    """
    if LLM_BACKEND == "ollama":
        return True
    return bool(ANTHROPIC_API_KEY)


def llm_backend_label() -> str:
    """Return a human-readable label for the active backend (used in CLI banner)."""
    if LLM_BACKEND == "ollama":
        return f"Ollama ({OLLAMA_MODEL})"
    return f"Claude ({LLM_MODEL})"

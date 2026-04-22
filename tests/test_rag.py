"""
tests/test_rag.py

Isolation tests for the AutoStream RAG pipeline.
Verifies that the retriever surfaces correct chunks for key business queries.
"""

import pytest
from rag.retriever import NoInfoSignal, RetrievalResult, get_vectorstore, retrieve


# ─── Shared fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def store():
    """Build (or load) the FAISS vector store once for all tests in this module."""
    return get_vectorstore()


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_combined_text(results) -> str:
    """Join all retrieved chunk contents into one searchable string."""
    if isinstance(results, NoInfoSignal):
        return ""
    return " ".join(r.document.page_content for r in results).lower()


# ─── Test 1: Pro Plan Pricing ─────────────────────────────────────────────────

def test_pro_plan_price(store):
    """The retriever must surface the $79/month Pro Plan price."""
    query = "What is the price of the Pro plan?"
    results = retrieve(query, k=3, store=store)

    assert not isinstance(results, NoInfoSignal), (
        "Expected real results for Pro plan query, got NoInfoSignal"
    )

    combined = _get_combined_text(results)
    assert "$79" in combined or "79" in combined, (
        f"Pro Plan price ($79) not found in retrieved chunks.\nChunks:\n{combined}"
    )


# ─── Test 2: Refund Policy ────────────────────────────────────────────────────

def test_refund_policy(store):
    """The retriever must surface the 7-day refund policy."""
    query = "Do you offer refunds?"
    results = retrieve(query, k=3, store=store)

    assert not isinstance(results, NoInfoSignal), (
        "Expected real results for refund query, got NoInfoSignal"
    )

    combined = _get_combined_text(results)
    assert "7" in combined or "refund" in combined or "money-back" in combined, (
        f"Refund policy not found in retrieved chunks.\nChunks:\n{combined}"
    )


# ─── Test 3: 4K on Basic Plan ─────────────────────────────────────────────────

def test_4k_not_on_basic(store):
    """The retriever must surface info clarifying 4K is not on the Basic Plan."""
    query = "Is 4K available on Basic?"
    results = retrieve(query, k=3, store=store)

    assert not isinstance(results, NoInfoSignal), (
        "Expected real results for 4K query, got NoInfoSignal"
    )

    combined = _get_combined_text(results)
    # Must mention either 4K or basic plan in context
    assert "4k" in combined or "basic" in combined or "720p" in combined, (
        f"4K / Basic plan info not found in retrieved chunks.\nChunks:\n{combined}"
    )


# ─── Test 4: Scores Are Non-Zero ──────────────────────────────────────────────

def test_scores_are_positive(store):
    """All returned RetrievalResult objects must have a positive similarity score."""
    query = "What plans does AutoStream offer?"
    results = retrieve(query, k=3, store=store)

    assert not isinstance(results, NoInfoSignal)
    for r in results:
        assert r.score > 0, f"Got a non-positive score: {r.score}"


# ─── Test 5: NoInfoSignal for Irrelevant Query ────────────────────────────────

def test_no_info_signal_for_irrelevant_query(store):
    """A completely off-topic query must return a NoInfoSignal (or low-scoring results)."""
    query = "What is the boiling point of nitrogen in Kelvin?"
    results = retrieve(query, k=3, store=store)

    # Either we get a NoInfoSignal, or every score is low
    if isinstance(results, NoInfoSignal):
        assert results.query == query
        assert "don't have that information" in results.message
    else:
        # L2-converted scores: anything below 0.5 is considered low similarity
        scores = [r.score for r in results]
        assert max(scores) < 0.5, (
            f"Off-topic query returned suspiciously high scores: {scores}"
        )


# ─── Test 6: Result Structure ─────────────────────────────────────────────────

def test_result_structure(store):
    """Every RetrievalResult must have a Document with page_content and metadata."""
    query = "What support is included in the Pro Plan?"
    results = retrieve(query, k=3, store=store)

    assert not isinstance(results, NoInfoSignal), (
        f"Expected real results for support query, got NoInfoSignal"
    )
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.document.page_content.strip(), "Document has empty page_content"
        assert "source" in r.document.metadata, "Document missing 'source' metadata"
        assert "chunk_id" in r.document.metadata, "Document missing 'chunk_id' metadata"

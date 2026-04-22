"""
rag/retriever.py

FAISS-backed semantic retriever for the AutoStream knowledge base.
Uses sentence-transformers (all-MiniLM-L6-v2) for free, local embeddings.
Persists the FAISS index to disk so it is only built once.
Includes a confidence threshold: queries with no chunks scoring >= 0.4
return a "no information" signal instead of hallucinating.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, NamedTuple, Optional

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag.loader import load_knowledge_base

# ─── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_DIR = Path(__file__).parent.parent / "faiss_index"
CONFIDENCE_THRESHOLD = 0.25   # L2-converted sim = 1/(1+dist); below this → "no info"


# ─── Result Types ─────────────────────────────────────────────────────────────

class RetrievalResult(NamedTuple):
    """Wraps a retrieved Document with its similarity score."""
    document: Document
    score: float        # Cosine similarity in [0, 1]; higher is better


class NoInfoSignal(NamedTuple):
    """Returned when all retrieved chunks fall below the confidence threshold."""
    query: str
    message: str = (
        "I don't have that information in my knowledge base. "
        "Please contact AutoStream support at support@autostream.io."
    )


# ─── Embedder ─────────────────────────────────────────────────────────────────

def _get_embedder() -> HuggingFaceEmbeddings:
    """Return a HuggingFace sentence-transformer embedder (local, no API key)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # Note: do NOT set normalize_embeddings here; FAISS uses L2 distance
        # internally, and we convert L2 → similarity in retrieve() below.
    )


# ─── FAISS Store ──────────────────────────────────────────────────────────────

def _build_vectorstore(chunks: List[Document], embedder: HuggingFaceEmbeddings) -> FAISS:
    """Build a FAISS vector store from document chunks and persist it."""
    print("[retriever] Building FAISS index from chunks …")
    store = FAISS.from_documents(chunks, embedder)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_INDEX_DIR))
    print(f"[retriever] Index saved to {FAISS_INDEX_DIR}")
    return store


def _load_vectorstore(embedder: HuggingFaceEmbeddings) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    print(f"[retriever] Loading existing FAISS index from {FAISS_INDEX_DIR}")
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True,
    )


def get_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Return a FAISS vector store, building it only when necessary.

    Args:
        force_rebuild: If True, always rebuild the index even if one exists.

    Returns:
        A loaded or freshly-built FAISS vector store.
    """
    embedder = _get_embedder()
    index_exists = (FAISS_INDEX_DIR / "index.faiss").exists()

    if index_exists and not force_rebuild:
        return _load_vectorstore(embedder)

    chunks = load_knowledge_base()
    return _build_vectorstore(chunks, embedder)


# ─── Public Retrieval API ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    k: int = 3,
    store: Optional[FAISS] = None,
    force_rebuild: bool = False,
) -> List[RetrievalResult] | NoInfoSignal:
    """Retrieve the top-k most relevant document chunks for a query.

    Applies a confidence threshold: if every returned chunk scores below
    CONFIDENCE_THRESHOLD, returns a NoInfoSignal instead of silently
    returning low-quality context that the LLM might hallucinate from.

    Args:
        query:         Natural language query string.
        k:             Number of chunks to retrieve (default 3).
        store:         Optional pre-loaded FAISS store (avoids re-loading).
        force_rebuild: Force index rebuild even if one already exists.

    Returns:
        List[RetrievalResult] if at least one chunk meets the threshold,
        otherwise a NoInfoSignal named tuple.
    """
    if store is None:
        store = get_vectorstore(force_rebuild=force_rebuild)

    # similarity_search_with_score returns (doc, L2_distance) tuples.
    # L2 distance is in [0, ∞); smaller = more similar.
    # We convert to a similarity in [0, 1] using: sim = 1 / (1 + distance)
    raw_results = store.similarity_search_with_score(query, k=k)

    if not raw_results:
        return NoInfoSignal(query=query)

    retrieval_results = [
        RetrievalResult(document=doc, score=round(1.0 / (1.0 + dist), 4))
        for doc, dist in raw_results
    ]

    best_score = max(r.score for r in retrieval_results)
    if best_score < CONFIDENCE_THRESHOLD:
        print(
            f"[retriever] All scores below threshold ({best_score:.3f} < "
            f"{CONFIDENCE_THRESHOLD}). Returning NoInfoSignal."
        )
        return NoInfoSignal(query=query)

    return retrieval_results


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is the price of the Pro plan?",
        "Do you offer refunds?",
        "Is 4K available on Basic?",
    ]

    store = get_vectorstore()

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = retrieve(q, k=3, store=store)

        if isinstance(result, NoInfoSignal):
            print(f"Response: {result.message}")
        else:
            for i, r in enumerate(result, 1):
                print(f"\n  [{i}] Score: {r.score:.4f}")
                print(f"      Source: {r.document.metadata.get('source', 'N/A')}")
                print(f"      Chunk: {r.document.page_content[:150]} …")

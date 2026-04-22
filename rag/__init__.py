"""RAG package — knowledge base loading and FAISS retrieval."""
from rag.loader import load_knowledge_base
from rag.retriever import NoInfoSignal, RetrievalResult, get_vectorstore, retrieve

__all__ = [
    "load_knowledge_base",
    "get_vectorstore",
    "retrieve",
    "RetrievalResult",
    "NoInfoSignal",
]

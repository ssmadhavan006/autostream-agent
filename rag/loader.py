"""
rag/loader.py

Loads and chunks the AutoStream knowledge base.
Supports both Markdown (.md) and JSON (.json) formats.
Returns a list of LangChain Document objects with metadata.
"""

import json
import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Constants ────────────────────────────────────────────────────────────────

KB_DIR = Path(__file__).parent
KB_MD_PATH = KB_DIR / "knowledge_base.md"
KB_JSON_PATH = KB_DIR / "knowledge_base.json"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_markdown(path: Path) -> List[Document]:
    """Load a markdown file as a single raw Document."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": str(path), "format": "markdown"})]


def _load_json(path: Path) -> List[Document]:
    """Load a JSON knowledge base (list of records) into Documents.

    Each JSON record becomes its own Document, where the content is built
    from the record's 'title' and 'content' fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    docs: List[Document] = []
    for record in records:
        content = f"{record.get('title', '')}\n{record.get('content', '')}"
        metadata = {
            "source": str(path),
            "format": "json",
            "id": record.get("id", ""),
            "category": record.get("category", ""),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


# ─── Chunker ──────────────────────────────────────────────────────────────────

def _chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Annotate each chunk with a unique chunk_id
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def load_knowledge_base(
    md_path: Path = KB_MD_PATH,
    json_path: Path = KB_JSON_PATH,
) -> List[Document]:
    """Load and chunk all available knowledge base files.

    Loads both the Markdown and JSON sources (if they exist), merges them,
    and returns a deduplicated, chunked list of Documents.

    Args:
        md_path:   Path to the Markdown knowledge base file.
        json_path: Path to the JSON knowledge base file.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    raw_docs: List[Document] = []

    if md_path.exists():
        print(f"[loader] Loading Markdown KB: {md_path}")
        raw_docs.extend(_load_markdown(md_path))
    else:
        print(f"[loader] Markdown KB not found, skipping: {md_path}")

    if json_path.exists():
        print(f"[loader] Loading JSON KB: {json_path}")
        raw_docs.extend(_load_json(json_path))
    else:
        print(f"[loader] JSON KB not found, skipping: {json_path}")

    if not raw_docs:
        raise FileNotFoundError(
            "No knowledge base files found. "
            f"Expected at: {md_path} or {json_path}"
        )

    chunks = _chunk_documents(raw_docs)
    print(f"[loader] Total chunks produced: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    chunks = load_knowledge_base()
    for c in chunks[:3]:
        print("---")
        print(f"Chunk ID: {c.metadata['chunk_id']} | Source: {c.metadata['source']}")
        print(c.page_content[:200])

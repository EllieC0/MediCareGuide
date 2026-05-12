"""
MediCareGuide RAG Module
====================
Offline retrieval-augmented generation using the CMS "Medicare & You 2026"
handbook (10050-medicare-and-you.pdf, 128 pages).

How it works
------------
1. BUILD (once, ~20-40 s on CPU):
   - Extract plain text from each PDF page with pypdf.
   - Skip pages 1-8 (cover / index — no prose content).
   - Split each page into overlapping 300-word chunks.
   - Embed every chunk with all-MiniLM-L6-v2 (22 MB, CPU-friendly).
   - Store vectors in a FAISS IndexFlatIP index.
   - Persist index + metadata to ~/.core.rag/ so the build runs once.

2. RETRIEVE (per query, ~100-250 ms on CPU):
   - Embed the user query (same model).
   - Cosine-search the FAISS index for the top-k most relevant chunks.
   - Return chunk text + source page numbers for prompt injection.

Public API
----------
    RAG_AVAILABLE          bool — False if any dependency is missing
    build_or_load_index()  → RAGIndex  (called once at app startup)
    retrieve(query, index, k=3) → list[dict]  each dict has 'text' and 'page'

Dependencies
------------
    pip install pypdf fastembed faiss-cpu
"""

from __future__ import annotations

import os
import re
import pickle
from pathlib import Path
from typing import NamedTuple

# ── Optional imports ──────────────────────────────────────────────────────
try:
    import numpy as np
    import faiss
    from pypdf import PdfReader
    from fastembed import TextEmbedding
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────

PDF_PATH      = Path(__file__).parent.parent / "data" / "10050-medicare-and-you.pdf"
CACHE_DIR     = Path.home() / ".core.rag"
INDEX_FILE    = CACHE_DIR / "index.faiss"
META_FILE     = CACHE_DIR / "meta.pkl"

# Embedding model — 22 MB, fast on CPU, strong semantic search
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Chunking parameters
WORDS_PER_CHUNK  = 300   # target words per chunk
OVERLAP_WORDS    = 60    # overlap between consecutive chunks on the same page
SKIP_PAGES       = 8     # PDF pages 1-8 are cover / index — skip them

# How many chunks to retrieve per query
DEFAULT_K = 3


# ── Data structures ───────────────────────────────────────────────────────

class Chunk(NamedTuple):
    text: str
    page: int          # 1-based PDF page number


class RAGIndex(NamedTuple):
    faiss_index: "faiss.Index"
    chunks: list[Chunk]
    model:  "SentenceTransformer"


# ── Text extraction ───────────────────────────────────────────────────────

def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Return [(page_number, page_text), ...] for every page after the
    front matter (page > SKIP_PAGES) that contains at least 80 characters
    of real text.
    """
    reader = PdfReader(str(pdf_path))
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        if i <= SKIP_PAGES:
            continue
        text = page.extract_text() or ""
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 80:
            pages.append((i, text))
    return pages


def _page_to_chunks(page_num: int, text: str) -> list[Chunk]:
    """
    Split a single page's text into overlapping word-window chunks.
    Returns one or more Chunk objects.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end        = min(start + WORDS_PER_CHUNK, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(text=chunk_text, page=page_num))
        if end == len(words):
            break
        start += WORDS_PER_CHUNK - OVERLAP_WORDS
    return chunks


def _build_chunks(pdf_path: Path) -> list[Chunk]:
    """Extract and chunk the entire PDF."""
    pages  = _extract_pages(pdf_path)
    chunks = []
    for page_num, text in pages:
        chunks.extend(_page_to_chunks(page_num, text))
    return chunks


# ── Embedding & indexing ──────────────────────────────────────────────────

def _embed(texts: list[str], model: "TextEmbedding") -> "np.ndarray":
    """Return L2-normalised embeddings (float32, shape [N, D])."""
    vecs = np.array(list(model.embed(texts)), dtype="float32")
    # fastembed already returns normalised vectors; normalise again to be safe
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (vecs / norms).astype("float32")


def _build_faiss_index(chunks: list[Chunk],
                       model:  "TextEmbedding") -> "faiss.Index":
    """Embed all chunks and build a FAISS IndexFlatIP (cosine via L2-norm)."""
    texts = [c.text for c in chunks]
    print(f"  [RAG] Embedding {len(texts)} chunks with {EMBED_MODEL}…")
    vecs  = _embed(texts, model)
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"  [RAG] Index built: {index.ntotal} vectors, dim={dim}")
    return index


# ── Public: build or load ─────────────────────────────────────────────────

def build_or_load_index(pdf_path: Path = PDF_PATH,
                        force_rebuild: bool = False) -> "RAGIndex | None":
    """
    Load the FAISS index from disk if it exists, otherwise build it from
    the PDF and save it.

    Returns a RAGIndex, or None if RAG_AVAILABLE is False or the PDF is
    missing.

    This function is designed to be called once at Streamlit startup
    (wrapped in @st.cache_resource so it runs only once per server process).
    """
    if not RAG_AVAILABLE:
        print("[RAG] Dependencies missing — RAG disabled.")
        return None
    if not pdf_path.exists():
        print(f"[RAG] PDF not found: {pdf_path} — RAG disabled.")
        return None

    model = TextEmbedding(f"sentence-transformers/{EMBED_MODEL}")

    # ── Load from cache ──────────────────────────────────────────────────
    if not force_rebuild and INDEX_FILE.exists() and META_FILE.exists():
        try:
            print("[RAG] Loading cached index…")
            faiss_index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "rb") as f:
                chunks = pickle.load(f)
            print(f"[RAG] Loaded {faiss_index.ntotal} vectors from cache.")
            return RAGIndex(faiss_index=faiss_index, chunks=chunks, model=model)
        except Exception as e:
            print(f"[RAG] Cache load failed ({e}), rebuilding…")

    # ── Build from PDF ───────────────────────────────────────────────────
    print(f"[RAG] Building index from {pdf_path.name}…")
    chunks      = _build_chunks(pdf_path)
    faiss_index = _build_faiss_index(chunks, model)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[RAG] Index saved to {CACHE_DIR}")

    return RAGIndex(faiss_index=faiss_index, chunks=chunks, model=model)


# ── Public: retrieve ──────────────────────────────────────────────────────

def retrieve(query: str,
             rag:   "RAGIndex",
             k:     int = DEFAULT_K) -> list[dict]:
    """
    Search for the k most relevant chunks for *query*.

    Returns a list of dicts:
        [{"text": "...", "page": 42}, ...]

    The caller can format these as a context block in the Gemma prompt:

        context = "\\n\\n".join(
            f"[Medicare & You 2026, p.{r['page']}] {r['text']}"
            for r in retrieve(query, rag)
        )
    """
    if not RAG_AVAILABLE or rag is None:
        return []
    query_vec = _embed([query], rag.model)           # shape [1, D]
    scores, idxs = rag.faiss_index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        chunk = rag.chunks[idx]
        results.append({"text": chunk.text, "page": chunk.page, "score": float(score)})
    return results


def format_context(results: list[dict], max_words: int = 400) -> str:
    """
    Format retrieved chunks into a compact context block for prompt injection.
    Truncates to roughly max_words total to keep the prompt size manageable.
    """
    if not results:
        return ""
    lines = []
    total = 0
    for r in results:
        words = r["text"].split()
        remaining = max_words - total
        if remaining <= 0:
            break
        snippet = " ".join(words[:remaining])
        lines.append(f"[Medicare & You 2026, p.{r['page']}] {snippet}")
        total += len(words[:remaining])
    return "\n\n".join(lines)

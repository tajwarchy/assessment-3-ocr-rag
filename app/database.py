"""
app/database.py
ChromaDB vector store — stores chunks, embeddings, and metadata.

Collection schema per chunk:
┌─────────────────┬────────────────────────────────────────────┐
│ Field           │ Description                                │
├─────────────────┼────────────────────────────────────────────┤
│ id              │ "{stored_filename}_p{page}_c{chunk_index}" │
│ embedding       │ 384-dim float vector (MiniLM)              │
│ document        │ Raw chunk text                             │
│ metadata        │ See below ↓                                │
└─────────────────┴────────────────────────────────────────────┘

Metadata fields:
┌──────────────────────┬─────────────────────────────────────┐
│ original_filename    │ Original uploaded filename          │
│ stored_filename      │ UUID filename saved to disk         │
│ page_number          │ Page the chunk came from (int)      │
│ chunk_index          │ Position of chunk within page (int) │
│ dominant_language    │ "bangla" | "english" | "mixed"      │
│ upload_date          │ ISO 8601 UTC string                  │
│ doc_type             │ "pdf" | "image"                      │
│ ocr_confidence       │ Avg Tesseract confidence (float)     │
└──────────────────────┴─────────────────────────────────────┘
"""

import chromadb
from chromadb.config import Settings

CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "documents"

# Module-level singleton
_client     = None  # chromadb.PersistentClient
_collection = None  # chromadb.Collection


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initializing on first call."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        # get_or_create so re-starts don't wipe data
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"[DB] ChromaDB ready. Collection '{COLLECTION_NAME}' "
              f"has {_collection.count()} chunks.")
    return _collection


def store_chunks(
    chunks:            list[dict],
    embeddings:        list[list[float]],
    original_filename: str,
    stored_filename:   str,
    dominant_language: str,
    upload_date:       str,
    doc_type:          str,
) -> int:
    """
    Insert chunks + embeddings into ChromaDB.
    Returns the number of chunks stored.
    """
    col = get_collection()

    ids, docs, metas, embeds = [], [], [], []

    for chunk, emb in zip(chunks, embeddings):
        chunk_id = (
            f"{stored_filename}"
            f"_p{chunk['page_number']}"
            f"_c{chunk['chunk_index']}"
        )
        ids.append(chunk_id)
        docs.append(chunk["text"])
        embeds.append(emb)
        metas.append({
            "original_filename": original_filename,
            "stored_filename":   stored_filename,
            "page_number":       chunk["page_number"],
            "chunk_index":       chunk["chunk_index"],
            "dominant_language": dominant_language,
            "upload_date":       upload_date,
            "doc_type":          doc_type,
            "ocr_confidence":    chunk["ocr_confidence"],
        })

    if ids:
        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)

    return len(ids)


def list_documents() -> list[dict]:
    """
    Return one summary entry per unique original_filename.
    """
    col = get_collection()
    if col.count() == 0:
        return []

    # Fetch all metadata (no embeddings needed)
    results = col.get(include=["metadatas"])
    seen, docs = set(), []

    for meta in results["metadatas"]:
        key = meta["stored_filename"]
        if key not in seen:
            seen.add(key)
            docs.append({
                "original_filename": meta["original_filename"],
                "stored_filename":   meta["stored_filename"],
                "dominant_language": meta["dominant_language"],
                "upload_date":       meta["upload_date"],
                "doc_type":          meta["doc_type"],
            })

    docs.sort(key=lambda x: x["upload_date"], reverse=True)
    return docs


def query_chunks(
    query_embedding: list[float],
    n_results:       int = 5,
    where           = None,
) -> list[dict]:
    """
    Vector similarity search with optional metadata filter.

    `where` is a ChromaDB filter dict, e.g.:
        {"dominant_language": "bangla"}
        {"$and": [{"dominant_language": "bangla"}, {"doc_type": "pdf"}]}
    """
    col = get_collection()
    if col.count() == 0:
        return []

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "metadata": meta,
            "score":    round(1 - dist, 4),  # cosine similarity (1 = identical)
        })

    return chunks
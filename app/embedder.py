"""
app/embedder.py
Loads a local multilingual sentence embedding model.

Model: paraphrase-multilingual-MiniLM-L12-v2
- Runs 100% locally via sentence-transformers
- Supports 50+ languages including Bangla and English
- 384-dimensional embeddings — lightweight and fast
- No API calls, no cost
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Module-level singleton — loaded once, reused across requests
_model = None  # SentenceTransformer


def get_model() -> SentenceTransformer:
    """Return the embedding model, loading it on first call."""
    global _model
    if _model is None:
        print(f"[EMBEDDER] Loading model: {MODEL_NAME} (first load only)...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[EMBEDDER] Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of float vectors (one per input text).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
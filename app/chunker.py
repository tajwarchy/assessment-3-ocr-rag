"""
app/chunker.py
Splits extracted OCR text into overlapping chunks for embedding.

Strategy:
- Chunk size  : 500 characters
- Overlap     : 50 characters
- Why chars?  : Bangla characters are multi-byte; token counts vary
                wildly across scripts, so character-based splitting is
                more predictable for a bilingual dataset.
- Overlap     : Preserves sentence context that falls across chunk
                boundaries, important for RAG retrieval accuracy.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# Separators tried in order — newlines first, then sentences, then words
_SEPARATORS = ["\n\n", "\n", "।", ".", " ", ""]
# "।" is the Bangla/Devanagari danda (sentence terminator)


def chunk_text(text: str) -> list[str]:
    """
    Split a block of text into overlapping chunks.
    Returns a list of non-empty string chunks.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_text(text)
    # Filter out chunks that are just whitespace or very short (noise)
    return [c.strip() for c in chunks if len(c.strip()) > 20]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunk each page's text separately, preserving page-level metadata.

    Input:  list of page dicts from ocr.process_file()
    Output: list of chunk dicts with metadata attached
    """
    all_chunks = []

    for page in pages:
        page_text = page.get("text", "")
        page_num  = page.get("page_number", 1)
        page_conf = page.get("avg_confidence", 0.0)

        chunks = chunk_text(page_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text":         chunk,
                "page_number":  page_num,
                "chunk_index":  i,
                "ocr_confidence": page_conf,
            })

    return all_chunks
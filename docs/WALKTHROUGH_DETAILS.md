# Technical Walkthrough

A component-by-component deep dive into how the Local OCR & Dynamic RAG System works internally.

---

## 1. System Overview

The pipeline has five stages, each implemented as an isolated module under `app/`:

```
Upload (FastAPI) → OCR (Tesseract) → Chunk (LangChain splitter)
→ Embed (sentence-transformers) → Store (ChromaDB)
                                          │
                                          ▼
                              Query → Vector Search + Metadata Filter
                                          │
                                          ▼
                              Context → Gemini API → Answer
```

Every stage runs locally except the final answer-generation call to the Gemini API. No document content, OCR text, embeddings, or search queries are sent to OCR or embedding vendors — only the final retrieved context + question are sent to Gemini for answer synthesis.

---

## 2. Document Ingestion (`app/main.py` → `/upload`)

The `/upload` endpoint accepts `multipart/form-data` file uploads via FastAPI's `UploadFile`. Accepted extensions are restricted to `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`.

Each upload is saved to disk under `uploads/` with a UUID-based filename (`uuid.uuid4().hex + extension`) to avoid filename collisions between different uploads of the same original filename. The original filename is preserved separately as metadata for display purposes.

---

## 3. Local OCR Extraction (`app/ocr.py`)

### Engine: Tesseract (`ben+eng`)

OCR runs via `pytesseract`, Python's wrapper around the Tesseract C++ engine. Two language packs are loaded simultaneously using the `lang="ben+eng"` parameter, allowing Tesseract to recognize Bangla and English characters in the same pass without needing to know in advance which language a given page or line is written in.

Configuration used: `--oem 3 --psm 3`
- `--oem 3`: uses Tesseract's LSTM-based recognition engine (its most accurate mode, versus the legacy engine)
- `--psm 3`: fully automatic page segmentation — Tesseract determines layout (paragraphs, columns, lines) without manual hints

### PDF Handling

PDFs are converted to images via `pdf2image.convert_from_path()` at 300 DPI before OCR, since Tesseract operates on raster images, not PDF text layers (this also means the system correctly handles *scanned* PDFs, not just text-based ones, which was a core requirement). 300 DPI was chosen as a balance between OCR accuracy (higher DPI generally improves character recognition) and processing time.

### Per-Page Confidence Scoring

For each page, `pytesseract.image_to_data()` is called (in addition to `image_to_string()`) to extract per-word confidence scores. These are averaged into a single `avg_confidence` value per page, stored alongside the extracted text and later propagated into each chunk's metadata — giving downstream consumers (or a human reviewer) a signal for how trustworthy a given chunk's OCR output is.

### Language Detection

`detect_language()` in `app/ocr.py` performs a lightweight heuristic: it counts characters falling in the Bangla Unicode block (`U+0980`–`U+09FF`) versus total alphabetic characters in the extracted text, then classifies the page as `"bangla"` (>60% Bangla characters), `"mixed"` (20–60%), or `"english"` (<20%). This avoids pulling in a heavier dedicated language-ID model for a binary/ternary classification that Unicode ranges already solve reliably for this script pair.

---

## 4. Chunking Strategy (`app/chunker.py`)

Uses LangChain's `RecursiveCharacterTextSplitter` with:
- **Chunk size:** 500 characters
- **Overlap:** 50 characters
- **Separator priority:** `["\n\n", "\n", "।", ".", " ", ""]`

### Why character-based, not token-based?

Token counts for the same visual length of text vary significantly between Bangla (a complex script with conjuncts and multi-byte Unicode) and English. Character-based chunking produces far more consistent, predictable chunk sizes across a bilingual corpus, which matters for keeping each chunk within a sensible context window for both the embedding model and the downstream LLM.

### Why this separator list?

The splitter tries each separator in order, falling back to the next if a chunk is still too large. Paragraph breaks (`\n\n`) and line breaks (`\n`) are tried first to preserve document structure. The Bangla sentence terminator `।` (danda) is included explicitly — without it, the splitter would have no language-aware way to find Bangla sentence boundaries, since Bangla doesn't reliably use the Latin period (`.`) the way English does. The plain period and space are kept as fallbacks for English and mixed content.

### Why overlap?

A 50-character overlap (10% of chunk size) ensures that a sentence or clause split across a chunk boundary still appears in full in at least one chunk, reducing the chance that a fact gets fragmented in a way that hurts retrieval.

### Per-Page Chunking

Chunking is performed independently per OCR'd page (`chunk_pages()`), rather than concatenating the entire document first. This preserves accurate `page_number` metadata per chunk, which is surfaced later in search results so users can trace an answer back to a specific page.

---

## 5. Embedding (`app/embedder.py`)

### Model: `paraphrase-multilingual-MiniLM-L12-v2`

Loaded once as a module-level singleton via `sentence-transformers`, avoiding the cost of reloading the model on every request. Key properties:

- **384-dimensional** output vectors — small enough for fast cosine similarity search at this dataset scale, while still carrying meaningful semantic signal
- **50+ language support**, including Bangla, trained with a paraphrase-identification objective so that semantically similar sentences land close together in vector space **regardless of language** — this is what allows a Bangla query to retrieve relevant English chunks (and vice versa) without any translation step
- **Fully local inference** — no embedding API calls, no per-query cost, no data leaving the machine

Both document chunks (at ingestion time) and user queries (at search time) are embedded with this same model, ensuring they live in the same vector space and are directly comparable.

---

## 6. Vector Storage (`app/database.py`)

### ChromaDB Schema

ChromaDB is used in **persistent local mode** (`PersistentClient(path="chroma_db")`), storing data on disk under `chroma_db/` rather than requiring a separate database server. A single collection (`"documents"`) holds every chunk from every uploaded document.

Each stored record consists of:

| Field | Type | Purpose |
|---|---|---|
| `id` | string | `{stored_filename}_p{page}_c{chunk_index}` — unique, traceable to source |
| `embedding` | float[384] | The chunk's semantic vector |
| `document` | string | The raw chunk text |
| `metadata.original_filename` | string | Human-readable filename for display |
| `metadata.stored_filename` | string | UUID filename on disk |
| `metadata.page_number` | int | Source page within the document |
| `metadata.chunk_index` | int | Position of chunk within that page |
| `metadata.dominant_language` | string | `"bangla"` \| `"english"` \| `"mixed"` |
| `metadata.upload_date` | string | ISO 8601 upload timestamp (human-readable) |
| `metadata.upload_timestamp` | float | Unix epoch seconds (numeric, used for range filtering) |
| `metadata.doc_type` | string | `"pdf"` \| `"image"` |
| `metadata.ocr_confidence` | float | Average Tesseract confidence for the source page |

The collection is configured with `metadata={"hnsw:space": "cosine"}`, meaning ChromaDB's underlying HNSW (Hierarchical Navigable Small World) index uses **cosine similarity** for nearest-neighbor search — appropriate for sentence-transformer embeddings, where direction matters more than magnitude.

### Why a separate numeric timestamp field?

ChromaDB's range query operators (`$gte`, `$lte`) only operate on numeric metadata fields, not strings — even ISO-formatted date strings. Storing `upload_timestamp` as a Unix epoch float alongside the human-readable `upload_date` string allows date-range filtering to work correctly while still keeping a readable date string for display in the UI.

---

## 7. Hybrid Search: Metadata Filtering + Vector Similarity (`app/rag.py`)

This is the core of the "dynamic hybrid search" requirement. The flow inside `search_and_answer()`:

1. **Embed the query** using the same local multilingual model used at ingestion time.
2. **Build a metadata filter** (`build_where_filter()`) from whichever manual filters the user supplied — language, document type, specific filename, and/or upload date range. Each active filter becomes a ChromaDB condition (e.g. `{"dominant_language": {"$eq": "bangla"}}`); if more than one filter is active, they're combined with a ChromaDB `$and` clause.
3. **Query ChromaDB** with both the query embedding *and* the `where` filter in a single call: `collection.query(query_embeddings=[...], where={...}, n_results=6)`.

### Why this is true hybrid search, not post-filtering

Critically, the `where` clause is passed **into** the same `.query()` call as the vector search — ChromaDB applies the metadata filter to restrict the candidate set *before* computing cosine similarity, rather than running an unrestricted similarity search and discarding non-matching results afterward. This means:
- Filtered searches are more efficient (similarity is computed only over the relevant subset)
- The `n_results` limit (top-6 chunks) is applied *within* the filtered subset, not diluted by irrelevant matches that get filtered out afterward

4. **Construct context** from the top retrieved chunks, each labeled with its source filename, page number, language, and similarity score, so the LLM (and the end user) can see provenance.
5. **Call Gemini** (`gemini-2.5-flash-lite`) with a prompt instructing it to answer strictly from the provided context, in the same language as the question, and to explicitly say when the answer isn't present in the retrieved context — reducing hallucination risk.
6. **Return** the generated answer alongside the raw source chunks (text, metadata, similarity score) so the frontend can display full provenance for every answer.

---

## 8. Frontend (`index.html`)

A single-page vanilla HTML/CSS/JS interface (no build step, no framework) communicating with the FastAPI backend via `fetch()`:

- **Upload panel:** drag-and-drop or click-to-browse file picker, posts to `/upload`, displays extracted text and per-document metadata chips on success.
- **Search panel:** a query input plus filter dropdowns/date inputs (language, document type, specific file, date range) that are serialized into the `/search` POST body. Results render the generated answer plus a list of source cards, each showing the originating file, page, language, and cosine similarity score.
- **Documents panel:** lists all currently stored documents (fetched from `/documents`, which reads unique filenames out of ChromaDB metadata) with a manual refresh button.

---

## 9. Containerization (`Dockerfile`, `docker-compose.yml`)

The Docker image is based on `python:3.11-slim` with Tesseract (`tesseract-ocr`, `tesseract-ocr-ben`, `tesseract-ocr-eng`) and `poppler-utils` installed at the OS level via `apt-get`, since these are system binaries that `pip` cannot install. Python dependencies are installed via `requirements.txt` in a separate layer before application code is copied in, so dependency installation is cached across rebuilds when only application code changes.

`docker-compose.yml` mounts `uploads/` and `chroma_db/` as host volumes, so uploaded documents and the vector database persist across container restarts rather than being wiped each time the container rebuilds.

---

## 10. End-to-End Data Flow Example

1. User uploads `document.pdf` (5 pages, mixed Bangla/English) → saved as `uploads/{uuid}.pdf`
2. Each of the 5 pages is OCR'd individually with Tesseract (`ben+eng`), producing per-page text + confidence scores
3. Each page's text is chunked independently (~500 chars, 50 char overlap) → e.g. 42 total chunks across 5 pages
4. All 42 chunks are embedded locally in a single batch via `sentence-transformers`
5. All 42 chunks + embeddings + metadata are inserted into ChromaDB in one `collection.add()` call
6. User searches "লিরা কি?" with `language=bangla` filter → query is embedded, ChromaDB returns the top 6 most similar chunks **restricted to chunks where `dominant_language == "bangla"`**
7. Those 6 chunks are formatted into a context block and sent to Gemini with the original question
8. Gemini's answer, plus the 6 source chunks (with scores and page numbers), are returned to the frontend and displayed
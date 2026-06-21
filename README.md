# Local OCR & Dynamic RAG System

A fully local, multilingual (Bangla + English) document processing pipeline with OCR-based ingestion and a hybrid Retrieval-Augmented Generation (RAG) search engine.

Built for **Assessment 3: Local OCR & Dynamic RAG System**.

---

## Features

- Upload scanned PDFs or images containing Bangla, English, or mixed text
- Fully local OCR using **Tesseract** — no data sent to external OCR APIs
- Automatic language detection per document (Bangla / English / Mixed)
- Text chunking with overlap, tuned for bilingual content
- Local multilingual embeddings via **sentence-transformers**
- Vector storage in **ChromaDB** (persisted locally on disk)
- Hybrid search: natural language query + manual metadata filters (language, document type, filename, date range)
- Answer generation via the **free-tier Gemini API**
- Simple browser-based UI — no frontend framework required

---

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│   Browser    │ ───▶ │   FastAPI     │ ───▶ │   Tesseract     │
│  (index.html)│      │   Backend     │      │   OCR (local)   │
└─────────────┘      └──────┬───────┘      └────────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │   Chunker     │  (RecursiveCharacterTextSplitter)
                      └──────┬───────┘
                              ▼
                      ┌──────────────┐
                      │  Embedder     │  (paraphrase-multilingual-MiniLM-L12-v2)
                      └──────┬───────┘
                              ▼
                      ┌──────────────┐
                      │  ChromaDB     │  (local persistent vector store)
                      └──────┬───────┘
                              │  vector search + metadata filter
                              ▼
                      ┌──────────────┐
                      │  Gemini API   │  (answer generation only)
                      └──────────────┘
```

**How metadata filtering works alongside vector search:** every chunk stored in ChromaDB carries metadata (`dominant_language`, `doc_type`, `original_filename`, `upload_date`, `page_number`, `ocr_confidence`). When a search request includes manual filters, they're converted into a ChromaDB `where` clause (e.g. `{"dominant_language": {"$eq": "bangla"}}`) and combined with `$and` if multiple filters are active. This `where` clause is applied **before** the cosine similarity search runs, so ChromaDB only computes similarity scores against the pre-filtered subset of chunks — this is a true hybrid search, not a post-filter on top of unrestricted results.

---

## Design Decisions & Trade-offs

### OCR Engine: Tesseract (not Surya)

Surya OCR was the original choice for potentially higher Bangla accuracy, but its Python API had breaking changes between versions and config incompatibilities with current `transformers` releases, causing repeated environment failures. **Tesseract** was chosen instead:

- **Pros:** extremely stable, zero version conflicts, native Bangla (`ben`) + English (`eng`) language packs, fast on CPU, no GPU required, decades of production use
- **Trade-offs:** Tesseract's accuracy on complex/cursive Bangla scripts, low-contrast scans, or unusual fonts is noticeably lower than modern transformer-based OCR (like Surya or vision-language models). On clean, well-scanned printed Bangla text, accuracy is solid (often 85-95% confidence per Tesseract's own scoring); on noisy or handwritten input, accuracy drops significantly.
- **Mitigation:** documents are processed at 300 DPI (`pdf2image` conversion) to maximize input quality before OCR, and both `ben` and `eng` language data are loaded simultaneously (`ben+eng`) so mixed-language lines are handled in a single pass.

### Chunking Strategy

- **Chunk size:** 500 characters, **overlap:** 50 characters
- **Why characters, not tokens:** Bangla is a multi-byte script and token counts vary unpredictably between Bangla and English text for the same visual length. Character-based chunking gives more consistent, predictable chunk sizes across a bilingual dataset.
- **Separators:** chunks are split first on paragraph breaks, then newlines, then the Bangla sentence terminator `।` (danda), then periods, then spaces — in that priority order — so chunk boundaries respect natural sentence structure in both languages wherever possible.
- **Overlap:** the 50-character overlap ensures that context spanning a chunk boundary (e.g. a sentence cut mid-way) isn't lost during retrieval.

### Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`

- Runs 100% locally via `sentence-transformers`, no API calls or cost
- Supports 50+ languages including Bangla, in a single shared embedding space — meaning a Bangla query can retrieve relevant English chunks and vice versa
- 384-dimensional vectors — lightweight, fast to compute and search, while still maintaining solid semantic accuracy for a system of this scale

### LLM for Answer Generation: Gemini 2.5 Flash Lite

Used only for the final answer-generation step (not for retrieval, which is fully local). Chosen because it's free-tier, fast, and capable of following multilingual instructions to answer in the same language as the question.

---

## Setup Guide (First-Time Setup)

### Prerequisites

- macOS (Apple Silicon supported) — for other OS, see Tesseract install notes below
- Python 3.11
- Homebrew (Mac package manager) — install from [brew.sh](https://brew.sh) if you don't have it

### 1. Clone the Repository

```bash
git clone https://github.com/tajwarchy/assessment-3-ocr-rag.git
cd assessment-3-ocr-rag
```

### 2. Install Tesseract OCR (system-level, not pip)

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang   # installs all language packs including Bangla (ben)
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng
```

**Windows:**
Download the installer from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki), run it, and ensure "Bangla" is checked under additional language data during install. Add the install folder to your system PATH.

**Verify installation:**
```bash
tesseract --list-langs
```
You should see `ben` and `eng` in the output.

### 3. Install `poppler` (required for PDF-to-image conversion)

**macOS:** `brew install poppler`
**Ubuntu/Debian:** `sudo apt install poppler-utils`
**Windows:** Download from [github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases), extract, add `bin/` folder to PATH.

### 4. Set Up Python Environment

```bash
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> First-time note: `sentence-transformers` will download the embedding model (~470MB) on its first run, and Tesseract's language data was already installed in Step 2. Both are one-time downloads.

### 5. Get a Free Gemini API Key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **"Create API Key"**
4. Copy the generated key

### 6. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and paste your Gemini key:
```
GEMINI_API_KEY=your_actual_key_here
```

### 7. Run the Application

**Option A — Local (Python venv):**
```bash
uvicorn app.main:app --reload --port 8000
```

**Option B — Docker (recommended for reproducibility):**

First-time Docker setup:
1. Download Docker Desktop from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) and install it
2. Open Docker Desktop once so the background engine is running
3. Verify: `docker --version`

Then, with your `.env` file already configured (Step 6 above):
```bash
docker compose up --build
```

This builds a container with Tesseract (+ Bangla language pack), poppler, and all Python dependencies pre-installed — no manual system setup needed inside the container. Uploaded files and the vector database persist locally via mounted volumes (`uploads/` and `chroma_db/`).

To stop: `docker compose down`
To view logs: `docker compose logs -f`

Open your browser to **http://localhost:8000** (same for both options)

---

## Usage

1. **Upload:** Drag a PDF or image (Bangla/English/mixed) into the upload box → click "Run OCR & Store". Watch your terminal for live processing logs.
2. **Search:** Type a natural language question (in Bangla or English) into the search box. Optionally set filters for language, document type, specific filename, or upload date range. Click "Ask".

   > **Note on date filtering:** the "From Date" / "To Date" filters refer to the **upload date** — i.e. when the document was processed and stored in the system — not any date mentioned inside the document's content. Extracting and filtering by in-document dates (e.g. an invoice or letter date) would require an additional LLM-based date extraction step, which is out of scope for this assessment but a natural extension.
3. **Review:** The generated answer appears along with the retrieved source chunks, their similarity scores, and originating page numbers.

---

## Project Structure

```
assessment-3-ocr-rag/
├── app/
│   ├── main.py          # FastAPI app & endpoints
│   ├── ocr.py            # Tesseract OCR processing
│   ├── chunker.py        # Text chunking logic
│   ├── embedder.py       # Local embedding model wrapper
│   ├── database.py       # ChromaDB vector store interface
│   └── rag.py             # RAG pipeline (retrieval + Gemini generation)
├── uploads/               # Uploaded files (gitignored)
├── chroma_db/              # Persistent vector store (gitignored)
├── index.html              # Frontend UI
├── test_ocr.py              # Standalone OCR test script
├── Dockerfile                # Container image definition
├── docker-compose.yml          # Container orchestration
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Endpoint      | Description                                      |
|--------|---------------|---------------------------------------------------|
| GET    | `/health`     | Health check + total chunks stored                |
| POST   | `/upload`     | Upload a document, run OCR, chunk, embed, store    |
| GET    | `/documents`  | List all stored documents                          |
| POST   | `/search`     | Hybrid RAG search with optional metadata filters   |

---

## Known Limitations

- Tesseract accuracy degrades on handwritten or heavily stylized Bangla fonts
- Language detection is based on Unicode character ratio, not a dedicated language-ID model — generally reliable for clearly Bangla or English text, less precise for heavily mixed content
- No OCR confidence-based rejection — low-confidence pages are still chunked and stored

##Demo Video

https://www.youtube.com/watch?v=dLnvq4oBE1w  
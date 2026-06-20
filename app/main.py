"""
app/main.py
FastAPI backend for Assessment 3 — Local OCR & RAG System.
"""

import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
from app.ocr      import process_file
from app.chunker  import chunk_pages
from app.embedder import embed_texts
from app.database import store_chunks, list_documents, get_collection
from app.rag      import search_and_answer

# ── Setup ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}

app = FastAPI(
    title="Local OCR & RAG System",
    description="Assessment 3 — Multilingual document processing pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/health")
def health():
    col = get_collection()
    return {
        "status": "ok",
        "message": "OCR & RAG backend is running.",
        "total_chunks_in_db": col.count(),
    }


# ── Upload → OCR → Chunk → Embed → Store ─────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path   = UPLOAD_DIR / unique_name
    doc_type    = "pdf" if suffix == ".pdf" else "image"
    upload_date = datetime.utcnow().isoformat()

    print(f"\n[UPLOAD] {file.filename} → {unique_name}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 1. OCR
    print("[OCR] Running local Tesseract OCR...")
    try:
        ocr_result = process_file(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    print(f"[OCR] Done. Language: {ocr_result['dominant_language']} | Pages: {ocr_result['total_pages']}")

    # 2. Chunk
    print("[CHUNK] Splitting text into chunks...")
    chunks = chunk_pages(ocr_result["pages"])
    print(f"[CHUNK] {len(chunks)} chunks created.")

    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from this document.")

    # 3. Embed
    print("[EMBED] Generating embeddings locally...")
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"[EMBED] {len(embeddings)} embeddings generated.")

    # 4. Store in ChromaDB
    print("[DB] Storing chunks in ChromaDB...")
    stored = store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        original_filename=file.filename,
        stored_filename=unique_name,
        dominant_language=ocr_result["dominant_language"],
        upload_date=upload_date,
        doc_type=doc_type,
    )
    print(f"[DB] Stored {stored} chunks.")

    return {
        "status":             "success",
        "original_filename":  file.filename,
        "stored_as":          unique_name,
        "upload_date":        upload_date,
        "total_pages":        ocr_result["total_pages"],
        "dominant_language":  ocr_result["dominant_language"],
        "chunks_stored":      stored,
        "full_text":          ocr_result["full_text"],
        "pages":              ocr_result["pages"],
    }


# ── List documents (from ChromaDB) ────────────────────────────────────────────
@app.get("/documents")
def documents():
    docs = list_documents()
    return {"total": len(docs), "documents": docs}


# ── Search request schema ─────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query:    str
    language: str = "any"   # "bangla" | "english" | "mixed" | "any"
    doc_type: str = "any"   # "pdf" | "image" | "any"
    filename: str = "any"   # original filename or "any"


# ── RAG Search endpoint ───────────────────────────────────────────────────────
@app.post("/search")
def search(req: SearchRequest):
    """
    Hybrid search: vector similarity + optional metadata filters.
    Returns a Gemini-generated answer grounded in uploaded documents.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    print(f"\n[SEARCH] Query: '{req.query}'")
    print(f"[SEARCH] Filters — language: {req.language} | doc_type: {req.doc_type} | file: {req.filename}")

    result = search_and_answer(
        query    = req.query,
        language = req.language,
        doc_type = req.doc_type,
        filename = req.filename,
    )

    print(f"[SEARCH] Returned {result['chunks_used']} chunks to Gemini.")
    return result
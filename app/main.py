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

from app.ocr import process_file

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

# Serve the frontend HTML at "/"
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "OCR & RAG backend is running."}


# ── Upload & OCR endpoint ─────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or image file.
    Runs Tesseract OCR locally and returns extracted text + metadata.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save uploaded file with a unique name to avoid collisions
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    print(f"\n[UPLOAD] Received: {file.filename} → saving as {unique_name}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run OCR locally
    print(f"[OCR] Starting local OCR on {unique_name}...")
    try:
        result = process_file(str(save_path))
    except Exception as e:
        # Clean up saved file on failure
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    print(f"[OCR] Done. Language: {result['dominant_language']} | Pages: {result['total_pages']}")

    return {
        "status": "success",
        "original_filename": file.filename,
        "stored_as": unique_name,
        "upload_date": datetime.utcnow().isoformat(),
        "total_pages": result["total_pages"],
        "dominant_language": result["dominant_language"],
        "full_text": result["full_text"],
        "pages": result["pages"],
    }


# ── List uploaded files ───────────────────────────────────────────────────────
@app.get("/documents")
def list_documents():
    """List all uploaded files in the uploads/ directory."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2),
                "uploaded_at": datetime.utcfromtimestamp(f.stat().st_mtime).isoformat(),
            })
    files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return {"total": len(files), "documents": files}
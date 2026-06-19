"""
app/ocr.py
Local OCR processing using Tesseract (ben+eng).
Handles both image files and PDFs.
"""

import time
from pathlib import Path
from PIL import Image
import pytesseract

TESS_LANG = "ben+eng"
TESS_CONFIG = "--oem 3 --psm 3"


def detect_language(text: str) -> str:
    """
    Naively detect dominant language in extracted text.
    Bangla unicode range: U+0980–U+09FF
    """
    bangla_chars = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "unknown"
    ratio = bangla_chars / total_alpha
    if ratio > 0.6:
        return "bangla"
    elif ratio > 0.2:
        return "mixed"
    return "english"


def ocr_image(img: Image.Image) -> dict:
    """Run Tesseract on a single PIL image. Returns text + confidence."""
    start = time.time()
    text = pytesseract.image_to_string(img, lang=TESS_LANG, config=TESS_CONFIG)

    data = pytesseract.image_to_data(
        img, lang=TESS_LANG, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    confidences = [
        int(c) for c in data["conf"]
        if str(c).lstrip("-").isdigit() and int(c) >= 0
    ]
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return {
        "text": text.strip(),
        "avg_confidence": avg_conf,
        "elapsed_s": round(time.time() - start, 2),
    }


def process_file(file_path: str) -> dict:
    """
    Main entry point. Accepts a path to a PDF or image.
    Returns structured OCR result with metadata.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    pages = []

    if suffix == ".pdf":
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), dpi=300)
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        images = [Image.open(str(path)).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    full_text_parts = []

    for i, img in enumerate(images):
        print(f"  [OCR] Page {i + 1}/{len(images)}...")
        result = ocr_image(img)
        pages.append({
            "page_number": i + 1,
            "text": result["text"],
            "avg_confidence": result["avg_confidence"],
            "elapsed_s": result["elapsed_s"],
        })
        full_text_parts.append(result["text"])

    full_text = "\n\n".join(full_text_parts)
    dominant_lang = detect_language(full_text)

    return {
        "filename": path.name,
        "total_pages": len(images),
        "dominant_language": dominant_lang,
        "full_text": full_text,
        "pages": pages,
    }
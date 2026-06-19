"""
Phase 1 - Tesseract OCR Test Script
Runs fully locally. Supports Bangla (ben) + English (eng).
Usage:
    python test_ocr.py                        # synthetic smoke test
    python test_ocr.py --file yourfile.pdf
    python test_ocr.py --file yourfile.png
"""

import argparse
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw
import pytesseract


# ── Language config ───────────────────────────────────────────────────────────
# "ben+eng" tells Tesseract to try both Bangla and English on every page
TESS_LANG = "ben+eng"
TESS_CONFIG = "--oem 3 --psm 3"
# oem 3 = LSTM engine (best accuracy)
# psm 3 = fully automatic page segmentation (default)


def load_images(file_path: str) -> list[Image.Image]:
    """Load image(s) from a PNG/JPG/PDF file."""
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    if path.suffix.lower() == ".pdf":
        print("[INFO] Detected PDF — converting pages to images...")
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), dpi=300)
        print(f"[INFO] Converted {len(images)} page(s).")
        return images
    else:
        print(f"[INFO] Loading image: {file_path}")
        return [Image.open(file_path).convert("RGB")]


def make_test_image() -> Image.Image:
    """Create a simple synthetic test image with English text."""
    img = Image.new("RGB", (900, 250), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 60),  "Hello! This is a local OCR test.",          fill=(0, 0, 0))
    draw.text((50, 110), "Assessment 3: Local OCR & RAG System",      fill=(0, 0, 0))
    draw.text((50, 160), "Tesseract is running fully offline.",        fill=(0, 0, 0))
    img.save("test_sample.png")
    print("[INFO] Saved synthetic test image → test_sample.png")
    return img


def run_ocr(images: list[Image.Image]) -> list[dict]:
    """Run Tesseract OCR on each image and return results."""
    results = []
    total_start = time.time()

    for i, img in enumerate(images):
        print(f"[INFO] Processing page {i + 1}/{len(images)}...")
        start = time.time()

        text = pytesseract.image_to_string(img, lang=TESS_LANG, config=TESS_CONFIG)

        # Get per-word confidence data
        data = pytesseract.image_to_data(
            img, lang=TESS_LANG, config=TESS_CONFIG,
            output_type=pytesseract.Output.DICT
        )
        confidences = [
            int(c) for c in data["conf"]
            if str(c).lstrip("-").isdigit() and int(c) >= 0
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        elapsed = time.time() - start
        results.append({
            "page": i + 1,
            "text": text.strip(),
            "avg_confidence": avg_conf,
            "word_count": len([w for w in text.split() if w]),
            "elapsed_s": round(elapsed, 2),
        })

    print(f"[INFO] Total OCR time: {time.time() - total_start:.2f}s\n")
    return results


def display_results(results: list[dict]):
    """Pretty-print OCR results."""
    for r in results:
        print(f"{'='*60}")
        print(f" PAGE {r['page']}")
        print(f"{'='*60}")

        if r["text"]:
            print(r["text"])
        else:
            print("[WARN] No text detected on this page.")

        print(f"\n[STATS] Words detected : {r['word_count']}")
        print(f"[STATS] Avg confidence : {r['avg_confidence']:.1f}%")
        print(f"[STATS] Time taken     : {r['elapsed_s']}s")


def check_tesseract():
    """Verify Tesseract is installed and Bangla lang pack is present."""
    try:
        langs = pytesseract.get_languages()
        print(f"[INFO] Tesseract version : {pytesseract.get_tesseract_version()}")
        print(f"[INFO] Available langs   : {', '.join(langs)}")
        if "ben" not in langs:
            print("[WARN] Bangla (ben) language pack NOT found!")
            print("       Run: brew install tesseract-lang")
        else:
            print("[INFO] Bangla (ben) language pack ✓")
    except Exception as e:
        print(f"[ERROR] Tesseract not found: {e}")
        print("        Run: brew install tesseract")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Tesseract OCR locally.")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a PNG/JPG/PDF file.")
    args = parser.parse_args()

    print("=" * 60)
    print(" TESSERACT OCR — LOCAL TEST")
    print("=" * 60)

    check_tesseract()
    print()

    if args.file:
        images = load_images(args.file)
    else:
        print("[INFO] No --file provided. Using synthetic test image.\n")
        images = [make_test_image()]

    results = run_ocr(images)
    display_results(results)

    print(f"\n{'='*60}")
    print(" SUCCESS — OCR pipeline is working locally.")
    print(f"{'='*60}")
    print("\nTip: Test with a real Bangla PDF:")
    print("     python test_ocr.py --file your_bangla_doc.pdf")


if __name__ == "__main__":
    main()
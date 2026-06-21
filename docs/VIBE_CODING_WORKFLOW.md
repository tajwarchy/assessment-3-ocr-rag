# Vibe Coding Workflow

This document describes how Claude (Anthropic's AI assistant) was used as a coding partner throughout the development of this project, including the iterative debugging process and key decisions made along the way.

---

## Overall Approach

The project was built using a **phase-by-phase, full-code-per-phase** workflow with Claude:

1. A complete project plan was generated upfront, breaking the assessment requirements into 6 sequential phases (environment setup, upload backend, chunking/embedding/storage, RAG search, frontend polish, Docker/finalization).
2. Each phase was approved before starting, then built and delivered in full — complete, working code for every file touched in that phase, not partial snippets.
3. After each phase, the code was run locally, real terminal output and errors were pasted back to Claude, and issues were debugged interactively before moving to the next phase.
4. Each completed phase ended with a git commit and push, keeping the repository history aligned with logical project milestones.

This kept the project structured and reviewable at every step, rather than generating the entire system in one pass.

---

## Where AI Assistance Was Used

- **Architecture planning:** Claude proposed the tech stack (Tesseract, FastAPI, ChromaDB, sentence-transformers, Gemini API) and the phase breakdown, which was reviewed and approved before implementation began.
- **Code generation:** All backend modules (`ocr.py`, `chunker.py`, `embedder.py`, `database.py`, `rag.py`, `main.py`), the frontend (`index.html`), Docker configuration, and documentation were drafted by Claude based on the assessment requirements.
- **Live debugging:** Real error tracebacks from the terminal (dependency conflicts, API errors, runtime exceptions) were pasted directly into the conversation, and Claude diagnosed root causes and supplied fixes iteratively.
- **Documentation:** The README, environment setup guide, and this docs folder were written collaboratively, translating implementation decisions into clear explanations.

---

## Key Pivots Made During Development

A vibe-coded project rarely goes in a straight line. Several real engineering decisions were made mid-build based on what actually happened when the code ran:

### 1. OCR Engine: Surya → Tesseract

The original plan used **Surya OCR** for potentially stronger Bangla accuracy. In practice, this hit a wall:
- Surya's Python API changed significantly between versions (v1 → v2 rework), breaking the originally planned `surya.ocr.run_ocr` import.
- Pinning to an older Surya version (`0.6.3`, then `0.4.3`) ran into `transformers` library incompatibilities (`KeyError: 'encoder'` during config loading).
- After two rounds of version-pinning attempts, the decision was made to **switch to Tesseract** — a more stable, battle-tested OCR engine with a native Bangla (`ben`) language pack and zero dependency conflicts.

This is a realistic example of a build-time trade-off: a "better on paper" model was abandoned in favor of a "boring but reliable" tool once it became clear the integration cost outweighed the accuracy benefit for this assessment's scope.

### 2. Gemini Model Name Drift

The RAG generation step initially used a Gemini model string (`gemini-2.5-flash-lite-preview-06-17`) that returned a `404 NOT_FOUND` from the API — the preview model had since been deprecated/renamed. This was fixed by switching to the stable `gemini-2.5-flash-lite` identifier, the same one used successfully in an earlier project (Assessment 1). This highlights a common pitfall with fast-moving LLM APIs: model identifiers can become stale even mid-project.

### 3. ChromaDB Metadata Filtering on Dates

Date-range filtering was first implemented by comparing ISO date strings directly (`{"upload_date": {"$gte": "2026-06-20"}}`), which failed at runtime — ChromaDB's `$gte`/`$lte` operators only support numeric fields, not strings. The fix was to store an additional numeric `upload_timestamp` field (Unix epoch) alongside the human-readable `upload_date` string, and filter on the numeric field instead. This is a good example of a filter design that looked correct on paper but only revealed its constraint once exercised against the real database engine.

### 4. Python Version Syntax Compatibility

Modern `X | None` union type hints (Python 3.10+ syntax) caused a `TypeError` when combined with how `chromadb`'s internals resolved types at import time. Rather than fight the interaction, type hints were simplified to plain comments/no annotation — a pragmatic fix that didn't affect functionality.

### 5. Frontend Bugs Caught via Browser Console

Two separate frontend issues (a button stuck in a disabled cursor state, and a missing `setStatus` helper function causing the Ask button to silently fail) were diagnosed by walking through the browser DevTools console output together, rather than guessing at the HTML/JS in isolation. This reinforced a debugging pattern used throughout the project: **always get the real error output before proposing a fix.**

---

## Debugging Pattern Used Throughout

A consistent loop was followed for every issue encountered:

1. Run the code locally
2. Paste the **exact** terminal output, traceback, or browser console error back into the conversation
3. Claude identifies the root cause from the actual error (not a guessed one)
4. A targeted fix is proposed and applied via the artifact update mechanism
5. Re-run and confirm before moving on

This "real error, real fix" loop avoided speculative debugging and kept each fix grounded in what was actually happening on the machine, rather than assumptions about what *should* be happening.

---

## Division of Responsibility

| Task | Primary Driver |
|---|---|
| Requirements interpretation & architecture choices | Claude (proposed), human (approved) |
| Code implementation | Claude |
| Running code, capturing real output/errors | Human |
| Diagnosing runtime errors from real tracebacks | Claude |
| Final testing & verification end-to-end | Human |
| Git commits & repository management | Human (commands provided by Claude) |
| Documentation | Claude (drafted), human (reviewed) |

---

## Reflection

This project illustrates a realistic AI-assisted development workflow: the AI did not produce a flawless system in one shot. Multiple components (OCR engine, LLM model identifier, date filtering logic, frontend event handling) required iteration based on real execution feedback. The value of the AI pairing was less about "getting it right the first time" and more about **rapid diagnosis and correction** once real-world behavior diverged from the plan — compressing what might otherwise be hours of solo debugging into a tight feedback loop.
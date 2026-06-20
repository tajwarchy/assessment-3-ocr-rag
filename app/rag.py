"""
app/rag.py
RAG pipeline — retrieves relevant chunks from ChromaDB
and generates an answer using Gemini API.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from app.embedder import embed_query
from app.database import query_chunks

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash-lite")

N_RESULTS = 6  # number of chunks to retrieve


def build_where_filter(
    language  = None,
    doc_type  = None,
    filename  = None,
    date_from = None,
    date_to   = None,
) -> dict | None:
    """
    Build a ChromaDB metadata filter dict from user-supplied filters.
    Only includes fields the user actually provided.
    """
    conditions = []

    if language and language != "any":
        conditions.append({"dominant_language": {"$eq": language}})
    if doc_type and doc_type != "any":
        conditions.append({"doc_type": {"$eq": doc_type}})
    if filename and filename != "any":
        conditions.append({"original_filename": {"$eq": filename}})
    if date_from:
        try:
            ts_from = datetime.fromisoformat(date_from).timestamp()
            conditions.append({"upload_timestamp": {"$gte": ts_from}})
        except ValueError:
            pass
    if date_to:
        try:
            ts_to = datetime.fromisoformat(date_to + "T23:59:59").timestamp()
            conditions.append({"upload_timestamp": {"$lte": ts_to}})
        except ValueError:
            pass

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_and_answer(
    query     : str,
    language  = None,
    doc_type  = None,
    filename  = None,
    date_from = None,
    date_to   = None,
) -> dict:
    """
    Full RAG pipeline:
    1. Embed the query locally
    2. Filter metadata (if filters supplied)
    3. Vector similarity search in ChromaDB
    4. Send retrieved chunks as context to Gemini
    5. Return answer + source chunks
    """

    # 1. Embed query
    q_embedding = embed_query(query)

    # 2. Build metadata filter
    where = build_where_filter(
        language=language, doc_type=doc_type, filename=filename,
        date_from=date_from, date_to=date_to,
    )

    # 3. Retrieve chunks
    chunks = query_chunks(
        query_embedding=q_embedding,
        n_results=N_RESULTS,
        where=where,
    )

    if not chunks:
        return {
            "answer": "No relevant content found. Try uploading a document first or adjusting your filters.",
            "sources": [],
            "filters_applied": where,
        }

    # 4. Build context string
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Source {i+1} | File: {meta['original_filename']} "
            f"| Page: {meta['page_number']} "
            f"| Lang: {meta['dominant_language']} "
            f"| Score: {chunk['score']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 5. Build prompt
    prompt = f"""You are a helpful multilingual assistant. Answer the user's question using ONLY the context provided below.
The context may contain Bangla, English, or mixed text. Answer in the same language as the question.
If the answer cannot be found in the context, say "I could not find relevant information in the uploaded documents."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    # 6. Call Gemini
    response = _model.generate_content(prompt)
    answer   = response.text.strip()

    return {
        "answer":          answer,
        "sources":         chunks,
        "filters_applied": where,
        "chunks_used":     len(chunks),
    }
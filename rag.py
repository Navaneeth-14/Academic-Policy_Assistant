import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks_from_pdf(filepath: str) -> list:
    """
    Extract text from a PDF and split into semantically meaningful chunks.

    Strategy:
    - Detect section headings (numbered like '1.', '2.1', or ALL CAPS lines)
    - Group text under each heading as one chunk
    - Fall back to paragraph splitting if no headings are found
    - Deduplicate near-identical chunks (removes repeated headers/footers)
    """
    import pdfplumber

    # ── Extract all text with page numbers ───────────────────────────────────
    full_lines = []   # list of (page_num, line_text)
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                full_lines.append((page_num, line.strip()))

    # ── Heading detection ─────────────────────────────────────────────────────
    heading_pattern = re.compile(
        r'^(\d+[\.\d]*\.?\s+[A-Z].{3,}|[A-Z][A-Z\s]{4,})$'
    )

    def is_heading(line: str) -> bool:
        return bool(heading_pattern.match(line)) and len(line) < 80

    # ── Build chunks under headings ───────────────────────────────────────────
    chunks = []
    current_section = "General"
    current_page    = 1
    current_text    = []

    for page_num, line in full_lines:
        if not line:
            continue
        if is_heading(line):
            if current_text:
                text_block = " ".join(current_text).strip()
                if len(text_block) > 40:
                    chunks.append({"section": current_section, "text": text_block})
            current_section = line
            current_page    = page_num
            current_text    = []
        else:
            current_text.append(line)

    # Flush last chunk
    if current_text:
        text_block = " ".join(current_text).strip()
        if len(text_block) > 40:
            chunks.append({"section": current_section, "text": text_block})

    # ── Fallback: if no headings detected, use paragraph splitting ────────────
    if len(chunks) <= 1:
        chunks = []
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                paragraphs = re.split(r"\n{2,}", text.strip())
                for para in paragraphs:
                    para = para.replace("\n", " ").strip()
                    if len(para) > 40:
                        chunks.append({"section": f"Page {page_num}", "text": para})

    # ── Deduplicate near-identical chunks (repeated headers/footers) ──────────
    seen  = set()
    deduped = []
    for chunk in chunks:
        # Use first 80 chars as fingerprint
        fingerprint = chunk["text"][:80].lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(chunk)

    return deduped


def load_chunks(filepath="data.txt"):
    """Parse data.txt into chunks with section labels."""
    chunks = []
    current_label = "GENERAL"
    current_text = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_text:
                    chunks.append({
                        "section": current_label,
                        "text": " ".join(current_text).strip()
                    })
                current_label = line[1:-1]
                current_text = []
            elif line:
                current_text.append(line)

    if current_text:
        chunks.append({"section": current_label, "text": " ".join(current_text).strip()})

    return chunks


def build_index(chunks):
    """Build a FAISS index from chunk embeddings."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query, chunks, index, top_k=3):
    """Return top_k most relevant chunks for the query."""
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "section": chunks[idx]["section"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })
    return results


# ── Chroma persistent vector store (optional, used for PDF uploads) ───────────

def build_chroma_index(chunks: list, collection_name: str = "policy_chunks"):
    """
    Build a persistent Chroma collection from chunks.
    Stored in ./chroma_db — survives app restarts.
    Returns the Chroma collection object.
    """
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection with same name to avoid stale data
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    texts    = [c["text"]    for c in chunks]
    sections = [c["section"] for c in chunks]
    ids      = [f"chunk_{i}" for i in range(len(chunks))]

    # Embed using the same sentence-transformer model
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"section": s} for s in sections]
    )
    return collection


def retrieve_from_chroma(query: str, collection, top_k: int = 5) -> list:
    """
    Retrieve top_k chunks from a Chroma collection.
    Returns same format as retrieve() for drop-in compatibility.
    """
    q_emb = model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks_out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Chroma cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score in [0, 1]
        score = 1 - (dist / 2)
        chunks_out.append({
            "section": meta["section"],
            "text":    doc,
            "score":   float(score)
        })
    return chunks_out

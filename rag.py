import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks_from_pdf(filepath: str) -> list:
    """Extract text from a PDF and split into paragraph-level chunks."""
    import pdfplumber
    chunks = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            # Split on blank lines or double newlines to get paragraphs
            paragraphs = re.split(r"\n{2,}", text.strip())
            for para in paragraphs:
                para = para.replace("\n", " ").strip()
                if len(para) > 40:   # skip very short fragments
                    chunks.append({
                        "section": f"Page {page_num}",
                        "text": para
                    })
    return chunks


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

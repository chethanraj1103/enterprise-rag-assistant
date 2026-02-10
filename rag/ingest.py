from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdfs(data_dir="data"):
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    metadata = []

    if not os.path.exists(data_dir):
        raise ValueError("data/ folder not found. Create it and add PDFs.")

    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(data_dir, file))
            text = " ".join(page.extract_text() or "" for page in reader.pages)

            if not text.strip():
                print(f"[WARN] No extractable text in {file}, skipping.")
                continue

            chunks = chunk_text(text)
            for c in chunks:
                all_chunks.append(c)
                metadata.append({"source": file})

    if len(all_chunks) == 0:
        raise ValueError("No text chunks found. Add a text-based PDF to data/.")

    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": metadata}, f)

    return len(all_chunks)

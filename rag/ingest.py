import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
import httpx

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def embed_texts(texts):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    embeddings = []

    for t in texts:
        r = httpx.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
            headers=headers,
            json={"inputs": t},
            timeout=60
        )
        emb = np.array(r.json()).mean(axis=0)
        embeddings.append(emb)

    return np.array(embeddings).astype("float32")

def ingest_pdfs(data_dir="data"):
    all_chunks = []
    metadata = []

    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            reader = PdfReader(os.path.join(data_dir, fname))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            chunks = chunk_text(text)

            all_chunks.extend(chunks)
            metadata.extend([{"source": fname}] * len(chunks))

    embeddings = embed_texts(all_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": metadata}, f)

    return len(all_chunks)

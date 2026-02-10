import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

model = SentenceTransformer(MODEL_NAME)
index = None
chunks = []
meta = []

def load_index():
    global index, chunks, meta
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            store = pickle.load(f)
        chunks = store["chunks"]
        meta = store["meta"]
        return True
    return False

def retrieve(query, k=3):
    global index, chunks, meta
    if index is None:
        ok = load_index()
        if not ok:
            return []

    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append({
            "text": chunks[idx],
            "source": meta[idx]["source"]
        })
    return results

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil

from rag.ingest import ingest_pdfs
from rag.qa import answer_question

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

@app.get("/", response_class=HTMLResponse)
def home():
    ui_path = BASE_DIR / "ui" / "index.html"
    if not ui_path.exists():
        return HTMLResponse(content="<h2>UI not found</h2>", status_code=404)
    return ui_path.read_text(encoding="utf-8")

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), x_role: str = Header(default="admin")):
    if x_role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Only admin can upload documents.")

    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_indexed = ingest_pdfs(str(data_dir))
    return {"status": "ok", "chunks_indexed": chunks_indexed}

@app.post("/ask")
def ask(question: str = Form(...)):
    answer, sources = answer_question(question)
    return {"answer": answer, "sources": sources}

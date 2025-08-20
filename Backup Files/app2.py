from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from memory_db import init_db
from agent import agent_reply, agent_reply_stream, build_bm25_index
from contextlib import asynccontextmanager
import json
import os
import re
import faiss
import numpy as np
from datetime import datetime
from openai import OpenAI
import shutil

# -------------------- Config --------------------
CHUNKS_FILE = "chunks.json"
WHATSAPP_EXPORT = "whatsapp_export.txt"
FAISS_INDEX_FILE = "whatsapp.index"
PROCESSED_FOLDER = "processed_exports"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- WhatsApp Export Parsing --------------------
def parse_whatsapp_txt(file_path):
    """Extract messages from a WhatsApp export file into a list of dicts with datetime."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex to capture: date, time, sender, message
    pattern = r"(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}\s?[ap]m)\s-\s([^:]+):\s(.+)"
    matches = re.findall(pattern, text)

    chunks = []
    for date_str, time_str, sender, msg in matches:
        msg = msg.strip()
        if msg:
            try:
                dt = datetime.strptime(f"{date_str} {time_str.replace(' ', '')}", "%d/%m/%Y%I:%M%p")
            except ValueError:
                dt = None
            chunks.append({"sender": sender, "text": msg, "datetime": dt})
    return chunks

def load_all_history():
    """Merge all archived exports into one chunk list."""
    all_chunks = []

    if os.path.exists(PROCESSED_FOLDER):
        for fname in sorted(os.listdir(PROCESSED_FOLDER)):
            if fname.endswith(".txt"):
                path = os.path.join(PROCESSED_FOLDER, fname)
                all_chunks.extend(parse_whatsapp_txt(path))

    if os.path.exists(CHUNKS_FILE):
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict) and "data" in existing_data:
                all_chunks.extend(existing_data["data"])
            elif isinstance(existing_data, list):
                all_chunks.extend(existing_data)
        except Exception:
            pass

    return all_chunks

def rebuild_indexes(chunks, last_updated):
    """Rebuild chunks.json and FAISS index from merged chunks."""
    data_to_save = {
        "_meta": {
            "last_updated": last_updated,
            "total_chunks": len(chunks)
        },
        "data": [{"sender": c["sender"], "text": c["text"]} for c in chunks]
    }
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    print(f"✅ Merged {len(chunks)} total chunks into {CHUNKS_FILE}")

    formatted_texts = [f"{c['sender']}: {c['text']}" for c in chunks]
    embeddings = []
    for text in formatted_texts:
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small", input=text
            ).data[0].embedding
            embeddings.append(emb)
        except Exception as e:
            print(f"❌ Failed to create embedding for '{text}': {e}")
            return

    if embeddings:
        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"✅ FAISS index saved to {FAISS_INDEX_FILE}")

def process_whatsapp_export():
    """Process new export, merge with history, sort chronologically, rebuild indexes, archive."""
    if not os.path.exists(WHATSAPP_EXPORT):
        return False

    export_mtime = os.path.getmtime(WHATSAPP_EXPORT)
    export_dt = datetime.fromtimestamp(export_mtime).isoformat()

    print(f"📄 Found {WHATSAPP_EXPORT}, processing...")
    new_chunks = parse_whatsapp_txt(WHATSAPP_EXPORT)

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    archive_name = f"whatsapp_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    archive_path = os.path.join(PROCESSED_FOLDER, archive_name)
    shutil.move(WHATSAPP_EXPORT, archive_path)
    print(f"📦 Archived {WHATSAPP_EXPORT} → {archive_path}")

    all_chunks = load_all_history()
    all_chunks.extend(new_chunks)

    seen = set()
    merged_chunks = []
    for chunk in all_chunks:
        key = (chunk["sender"], chunk["text"], chunk.get("datetime"))
        if key not in seen:
            seen.add(key)
            merged_chunks.append(chunk)

    merged_chunks.sort(key=lambda c: c.get("datetime") or datetime.max)

    rebuild_indexes(merged_chunks, export_dt)
    return True

# -------------------- Lifespan Events --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    process_whatsapp_export()

    if not os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"_meta": {"last_updated": None, "total_chunks": 0}, "data": []}, f)
        print(f"⚠️ {CHUNKS_FILE} not found — created empty file.")

    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks_file_content = json.load(f)
            chunks = chunks_file_content["data"] if isinstance(chunks_file_content, dict) else chunks_file_content
        build_bm25_index(chunks)
        print(f"✅ Loaded {len(chunks)} chunks into BM25 index")
    except Exception as e:
        print(f"⚠️ Failed to load BM25 index: {e}")

    yield
    print("🛑 Application shutting down...")

# -------------------- App Instance --------------------
app = FastAPI(lifespan=lifespan)

# -------------------- Routes --------------------
@app.post("/reply")
def reply(chat_id: str = Form(...), message: str = Form(...)):
    response_text = agent_reply(message, chat_id)
    return {"reply": response_text}

@app.post("/reply_stream")
def reply_stream(chat_id: str = Form(...), message: str = Form(...)):
    return StreamingResponse(
        agent_reply_stream(message, chat_id),
        media_type="text/plain"
    )

# -------------------- Main Entry --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
